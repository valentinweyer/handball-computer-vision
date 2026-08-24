"""Per-video prototype calibration for teams and referee rejection."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .model import (
    MAX_TORSO_CONTAMINATION,
    crop_quality,
    jersey_color_features,
    torso_boxes,
    torso_contamination,
)

CALIBRATION_SCHEMA_VERSION = 1
TEAM_A = 0
TEAM_B = 1
ROLE_FIELD = 0
ROLE_REFEREE = 1
CLASS_NAMES = {TEAM_A: "team_a", TEAM_B: "team_b", 2: "referee"}


def spatial_jersey_features(crops) -> np.ndarray:
    """Whole-crop plus spatial Lab/HSV descriptors for jersey robustness."""
    features = []
    for crop in crops:
        if crop is None or crop.size == 0:
            features.append(np.zeros(62 * 4, dtype=np.float32))
            continue
        height, width = crop.shape[:2]
        x1, x2 = round(width * 0.18), round(width * 0.82)
        y_mid = max(1, height // 2)
        regions = [
            crop,
            crop[:, x1:max(x1 + 1, x2)],
            crop[:y_mid],
            crop[y_mid:],
        ]
        features.append(
            np.concatenate(jersey_color_features(regions)).astype(np.float32)
        )
    return (
        np.stack(features)
        if features
        else np.empty((0, 62 * 4), dtype=np.float32)
    )


class PrototypeTeamCalibrator:
    """Cheap per-detection team probabilities from two or three clicked seeds.

    The seed crops anchor semantic team A/B and optionally referee. Clean
    unlabeled crops from the same video expand each seed into several local
    prototypes, retaining per-video adaptation without trusting track IDs.
    """

    def __init__(
        self, scaler, reducer, projection_scaler,
        prototypes: dict[int, np.ndarray], metadata: dict | None = None,
    ) -> None:
        self.scaler = scaler
        self.reducer = reducer
        self.projection_scaler = projection_scaler
        self.prototypes = {
            int(key): np.asarray(value, dtype=float)
            for key, value in prototypes.items()
        }
        self.metadata = metadata or {}

    @classmethod
    def fit(
        cls,
        pool_crops,
        seed_crops: dict[int, list[np.ndarray]],
        seed: int = 0,
        metadata: dict | None = None,
    ) -> "PrototypeTeamCalibrator":
        if TEAM_A not in seed_crops or TEAM_B not in seed_crops:
            raise ValueError("one prototype is required for both team A and team B")
        pool_crops = [crop for crop in pool_crops if crop_quality(crop).accepted]
        if len(pool_crops) < 8:
            raise ValueError(f"need at least 8 clean pool crops, got {len(pool_crops)}")
        labels = sorted(seed_crops)
        all_seed_crops = [crop for label in labels for crop in seed_crops[label]]
        if any(not crops for crops in seed_crops.values()):
            raise ValueError("every configured class needs at least one seed crop")

        pool_features = spatial_jersey_features(pool_crops)
        seed_features = spatial_jersey_features(all_seed_crops)
        scaler = StandardScaler().fit(np.vstack([pool_features, seed_features]))
        pool_scaled = scaler.transform(pool_features)
        seed_scaled = scaler.transform(seed_features)
        component_count = min(12, len(pool_scaled) - 1, pool_scaled.shape[1])
        reducer = PCA(n_components=component_count, random_state=seed).fit(
            np.vstack([pool_scaled, seed_scaled])
        )
        pool_projection = reducer.transform(pool_scaled)
        seed_projection = reducer.transform(seed_scaled)
        # Preserve PCA's explained-variance weighting. Scaling every component
        # to unit variance makes tiny histogram-bin noise as important as the
        # dominant jersey-colour axes, which is especially harmful with only
        # one clicked seed per class. Centering is sufficient here.
        projection_scaler = StandardScaler(with_std=False).fit(pool_projection)
        pool_projection = projection_scaler.transform(pool_projection)
        seed_projection = projection_scaler.transform(seed_projection)

        seed_by_label = {}
        cursor = 0
        for label in labels:
            count = len(seed_crops[label])
            seed_by_label[label] = seed_projection[cursor:cursor + count]
            cursor += count

        seed_centers = np.stack([
            seed_by_label[label].mean(axis=0) for label in labels
        ])
        seed_distances = np.linalg.norm(
            pool_projection[:, None, :] - seed_centers[None, :, :], axis=2
        )
        order = np.argsort(seed_distances, axis=1)
        nearest = order[:, 0]
        first = seed_distances[np.arange(len(pool_projection)), order[:, 0]]
        second = seed_distances[np.arange(len(pool_projection)), order[:, 1]]
        margin = np.divide(
            second - first,
            second + first,
            out=np.zeros_like(first),
            where=(second + first) > 1e-8,
        )

        prototypes = {}
        for label_index, label in enumerate(labels):
            confident = pool_projection[
                (nearest == label_index) & (margin >= 0.15)
            ]
            if len(confident) > 120:
                distances = seed_distances[
                    (nearest == label_index) & (margin >= 0.15), label_index
                ]
                confident = confident[np.argsort(distances)[:120]]
            combined = np.vstack([seed_by_label[label], confident])
            cluster_count = min(3, max(1, len(combined) // 12 + 1))
            if cluster_count == 1:
                centers = combined.mean(axis=0, keepdims=True)
            else:
                centers = KMeans(
                    n_clusters=cluster_count, random_state=seed, n_init=20
                ).fit(combined).cluster_centers_
            prototypes[label] = np.vstack([seed_by_label[label], centers])

        return cls(
            scaler, reducer, projection_scaler, prototypes, metadata
        )

    def _project(self, crops) -> np.ndarray:
        features = self.scaler.transform(spatial_jersey_features(crops))
        projection = self.reducer.transform(features)
        return self.projection_scaler.transform(projection)

    def distances(self, crops) -> dict[int, np.ndarray]:
        projection = self._project(crops)
        return {
            label: np.linalg.norm(
                projection[:, None, :] - values[None, :, :], axis=2
            ).min(axis=1)
            for label, values in self.prototypes.items()
        }

    def predict_crops(self, crops):
        """Return p(team B), team certainty, role id, and role certainty."""
        if not crops:
            empty = np.empty(0, dtype=float)
            return empty, empty, np.empty(0, dtype=int), empty
        distance = self.distances(crops)
        distance_a = distance[TEAM_A]
        distance_b = distance[TEAM_B]
        denominator = distance_a + distance_b
        probability_b = np.divide(
            distance_a,
            denominator,
            out=np.full(len(crops), 0.5, dtype=float),
            where=denominator > 1e-8,
        )
        team_certainty = np.abs(2.0 * probability_b - 1.0)
        role = np.full(len(crops), ROLE_FIELD, dtype=int)
        role_certainty = np.zeros(len(crops), dtype=float)
        if 2 in distance:
            referee_distance = distance[2]
            team_distance = np.minimum(distance_a, distance_b)
            role_certainty = np.divide(
                team_distance - referee_distance,
                team_distance + referee_distance,
                out=np.zeros(len(crops), dtype=float),
                where=(team_distance + referee_distance) > 1e-8,
            )
            role_certainty = np.clip(role_certainty, 0.0, 1.0)
            role[role_certainty >= 0.15] = ROLE_REFEREE
        return (
            np.clip(probability_b, 0.0, 1.0),
            np.clip(team_certainty, 0.0, 1.0),
            role,
            role_certainty,
        )

    @staticmethod
    def _crops_and_quality(
        frame_rgb: np.ndarray, boxes_xyxy: np.ndarray,
        context_boxes_xyxy: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
        crops = [
            sv.crop_image(frame_rgb, box) for box in torso_boxes(boxes)
        ]
        quality = np.array([
            crop_quality(crop).score for crop in crops
        ], dtype=float)
        contamination = torso_contamination(boxes, context_boxes_xyxy)
        quality *= np.clip(
            1.0 - contamination / MAX_TORSO_CONTAMINATION, 0.0, 1.0
        )
        return crops, quality

    def predict_frame(
        self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray,
        context_boxes_xyxy: np.ndarray | None = None,
    ):
        crops, quality = self._crops_and_quality(
            frame_rgb, boxes_xyxy, context_boxes_xyxy
        )
        probability_b, certainty, role, role_certainty = self.predict_crops(crops)
        return probability_b, certainty, role, role_certainty, quality

    def save(self, path: Path | str) -> None:
        with open(path, "wb") as file:
            pickle.dump({
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "scaler": self.scaler,
                "reducer": self.reducer,
                "projection_scaler": self.projection_scaler,
                "prototypes": self.prototypes,
                "metadata": self.metadata,
            }, file)

    @classmethod
    def load(cls, path: Path | str) -> "PrototypeTeamCalibrator":
        with open(path, "rb") as file:
            state = pickle.load(file)
        if state.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(f"stale team calibration: {path}")
        return cls(
            state["scaler"], state["reducer"],
            state["projection_scaler"], state["prototypes"],
            state.get("metadata"),
        )


class CalibratedTeamObserver:
    """IdentityManager-compatible appearance + calibrated team observer."""

    def __init__(self, appearance_model, calibrator: PrototypeTeamCalibrator):
        self.appearance_model = appearance_model
        self.calibrator = calibrator

    def observe(
        self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray,
        context_boxes_xyxy: np.ndarray | None = None,
    ):
        crops, quality = self.calibrator._crops_and_quality(
            frame_rgb, boxes_xyxy, context_boxes_xyxy
        )
        embeddings = self.appearance_model.extract_features(crops)
        probability_b, certainty, _role, _role_confidence = (
            self.calibrator.predict_crops(crops)
        )
        teams = (probability_b >= 0.5).astype(int)
        return embeddings, teams, certainty, quality

    def extract_features(self, crops):
        return self.appearance_model.extract_features(crops)
