"""Per-video, label-free team discovery.

The model learns two anonymous team clusters independently for every video.
Clean, centered torso crops provide robust Lab/HSV jersey histograms; SigLIP
embeddings remain available for player re-identification and are used as a
guarded fallback only when color is ambiguous and both fits agree.

Predictions carry explicit crop-quality and inter-player-overlap weights.
Downstream managers aggregate this evidence over a short neutral tracklet, so
an unusable or contaminated frame cannot permanently assign the wrong team.
Goalkeepers remain a detector class and are excluded from the two-team fit.
"""
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sports import TeamClassifier

# ── crop geometry ─────────────────────────────────────────────────────────
# "waist_v1" (default): the original centre-anchored 0.4-scale crop. "torso":
# full width, top-anchored [TORSO_TOP_FRAC, TORSO_BOTTOM_FRAC] of box height --
# looks sharper and jersey-centred in a still frame, measured worse on the live
# pipeline (see module docstring). Kept available, not deleted, since the
# failure mode (pose sensitivity) is footage-dependent -- less dynamic footage
# might favour it.
CROP_KIND = "waist_v1"
TORSO_TOP_FRAC = 0.10
TORSO_BOTTOM_FRAC = 0.60

# Scale factors for the "waist_v1" crop (centre-anchored, sv.scale_boxes-style):
# each axis spans [50-50*f, 50+50*f]% of box height/width independently.
# 0.4/0.4 is the original, isotropic value -- and, after two follow-up attempts
# to enlarge it, still the best measured on this clip:
#
#     W     H     team_purity_%
#     0.4   0.4   92.79% / 86.56%   <- default, two independent runs
#     0.4   0.55  80.40%
#     0.7   0.7   55.06%
#
# 0.7/0.7 widened from 40% to 70% of box WIDTH, which isn't tightly fit to the
# player and pulled in an adjacent player's kit in a scrum -- visually
# confirmed in side-by-side crops. Isolating height alone (0.4/0.55, width
# held at the proven-safe 0.4) ruled out that neighbour-bleed mechanism and
# still lost ~6-12pp: HEIGHT growth is centred too, so it adds as much head/
# scalp as leg, not more jersey, and that dilution alone is enough to hurt.
# Conclusion: any enlargement past 0.4/0.4 tried so far costs purity on this
# clip -- it already sits near a local optimum for this detector's box
# tightness. Left switchable (not hardcoded to 0.4) in case a future footage
# source has looser or tighter detector boxes where this doesn't hold.
CENTERED_CROP_SCALE_W = 0.4
CENTERED_CROP_SCALE_H = 0.4

# "kmeans" (default) or "gmm". GMM gives genuine predict_proba probabilities at
# a measured ~9.6pp team_purity_% cost on this clip (see module docstring) --
# a real, usable trade for a use case that needs calibrated per-detection
# probabilities rather than raw accuracy.
CLUSTER_MODEL_KIND = "kmeans"

# ── model/cache versioning ────────────────────────────────────────────────
# Bumped whenever crop geometry or model architecture changes, so a cache fit
# under the old geometry is never silently paired with new inputs -- it would
# produce plausible-looking but wrong predictions instead of an obvious error.
MODEL_SCHEMA_VERSION = 8

# Clean-crop benchmarks favour jersey colour over unconditional fusion:
# HAN/BER improves substantially over the old visual model; on Felix, colour
# scores 98.59%, visual 88.73%, and fixed fusion 97.18%. SigLIP remains useful
# for re-ID and as a guarded fallback when colour is genuinely ambiguous.
COLOR_PCA_COMPONENTS = 2
MIN_COLOR_CONFIDENCE = 0.08
MIN_VISUAL_FALLBACK_CONFIDENCE = 0.25
MIN_VISUAL_COLOR_AGREEMENT = 0.70

# Reject only clearly unusable observations. Borderline crops retain a
# continuous quality weight for temporal aggregation.
MIN_CROP_SIDE = 10
MIN_CROP_AREA = 400
MIN_CROP_CONTRAST = 12.0
MIN_CROP_SHARPNESS = 20.0
MAX_TORSO_CONTAMINATION = 0.25

# Symmetric evidence prior: one noisy first frame cannot lock a team, while a
# few consistent observations can.
TEAM_EVIDENCE_PRIOR = 0.5
MIN_STABLE_TEAM_CONFIDENCE = 0.70

_GEOMETRY_SIGNATURE = (
    CROP_KIND, TORSO_TOP_FRAC, TORSO_BOTTOM_FRAC,
    CENTERED_CROP_SCALE_W, CENTERED_CROP_SCALE_H, CLUSTER_MODEL_KIND,
    COLOR_PCA_COMPONENTS,
)

# KMeans' distance-margin proxy and GMM's predict_proba are on different scales
# (proxy: 0-1, ambiguous near 0; posterior: always >= 0.5 for 2 components).
# The KMeans gate is measured on independent RF-DETR crops from FelixClaar and
# Han-Ber4: 0.30 retains 70/71 and 58/64 labeled reads respectively, with 100%
# accuracy in both retained sets. Lower-confidence reads remain available as
# "unknown" observations but cannot poison team votes or gated association.
MIN_TEAM_VOTE_CONFIDENCE = 0.30 if CLUSTER_MODEL_KIND == "kmeans" else 0.75

# UMAP's default n_neighbors=15 needs at least that many samples to behave --
# comfortably met by the multi-frame fit (typically 100+ crops).
UMAP_N_COMPONENTS = 3


def _build_cluster_model(seed: int):
    if CLUSTER_MODEL_KIND == "kmeans":
        return KMeans(n_clusters=2, random_state=seed, n_init=10)
    if CLUSTER_MODEL_KIND == "gmm":
        return GaussianMixture(n_components=2, covariance_type="full", random_state=seed, n_init=5)
    raise ValueError(f"unknown CLUSTER_MODEL_KIND: {CLUSTER_MODEL_KIND!r}")


def torso_boxes(xyxy: np.ndarray) -> np.ndarray:
    """Crop geometry for team classification, per CROP_KIND (see module docstring)."""
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    if CROP_KIND == "waist_v1":
        centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2
        scale = np.array([CENTERED_CROP_SCALE_W, CENTERED_CROP_SCALE_H])
        sizes = (xyxy[:, 2:] - xyxy[:, :2]) * scale
        return np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1)
    h = xyxy[:, 3] - xyxy[:, 1]
    y1 = xyxy[:, 1] + h * TORSO_TOP_FRAC
    y2 = xyxy[:, 1] + h * TORSO_BOTTOM_FRAC
    return np.stack([xyxy[:, 0], y1, xyxy[:, 2], y2], axis=1)


@dataclass(frozen=True)
class CropQuality:
    accepted: bool
    score: float
    width: int
    height: int
    contrast: float
    sharpness: float


def crop_quality(crop_rgb: np.ndarray) -> CropQuality:
    """Cheap, tracker-independent evidence quality for one torso crop."""
    if crop_rgb is None or crop_rgb.size == 0:
        return CropQuality(False, 0.0, 0, 0, 0.0, 0.0)
    height, width = crop_rgb.shape[:2]
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    area = width * height
    accepted = (
        min(width, height) >= MIN_CROP_SIDE
        and area >= MIN_CROP_AREA
        and contrast >= MIN_CROP_CONTRAST
        and sharpness >= MIN_CROP_SHARPNESS
    )
    if not accepted:
        return CropQuality(False, 0.0, width, height, contrast, sharpness)
    size_score = float(np.clip(np.sqrt(area / 1000.0), 0.0, 1.0))
    contrast_score = float(np.clip(contrast / 35.0, 0.0, 1.0))
    sharpness_score = float(
        np.clip(np.log1p(sharpness) / np.log1p(300.0), 0.0, 1.0)
    )
    score = float((size_score * contrast_score * sharpness_score) ** (1.0 / 3.0))
    return CropQuality(True, score, width, height, contrast, sharpness)


def torso_contamination(
    boxes_xyxy: np.ndarray, context_boxes_xyxy: np.ndarray | None = None,
) -> np.ndarray:
    """Fraction of each target torso covered by another player box."""
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    context = (
        boxes
        if context_boxes_xyxy is None
        else np.asarray(context_boxes_xyxy, dtype=float).reshape(-1, 4)
    )
    torsos = torso_boxes(boxes)
    result = np.zeros(len(boxes), dtype=float)
    same_array = context_boxes_xyxy is None
    for i, (box, torso) in enumerate(zip(boxes, torsos)):
        area = max((torso[2] - torso[0]) * (torso[3] - torso[1]), 1.0)
        self_index = i if same_array else None
        if not same_array and len(context):
            iou = sv.box_iou_batch(box[None], context)[0]
            if iou.max() > 0.5:
                self_index = int(iou.argmax())
        for j, other in enumerate(context):
            if j == self_index:
                continue
            x1, y1 = max(torso[0], other[0]), max(torso[1], other[1])
            x2, y2 = min(torso[2], other[2]), min(torso[3], other[3])
            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            result[i] = max(result[i], intersection / area)
    return result


def _jersey_color_feature(crop: np.ndarray, valid_mask=None) -> np.ndarray:
    """Lab/HSV jersey descriptor, optionally restricted to selected pixels."""
    if crop is None or crop.size == 0:
        return np.zeros(62, dtype=np.float32)
    resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    selected = np.ones((32, 32), dtype=bool)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=np.uint8)
        if mask.shape != crop.shape[:2]:
            raise ValueError(
                "valid mask must have the same height and width as its crop: "
                f"{mask.shape} != {crop.shape[:2]}"
            )
        selected = cv2.resize(
            mask, (32, 32), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        if not selected.any():
            return np.zeros(62, dtype=np.float32)

    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    values = []
    for channel, bins, value_range in (
        (lab[:, :, 1], 12, (0, 256)),
        (lab[:, :, 2], 12, (0, 256)),
        (hsv[:, :, 0], 12, (0, 180)),
        (hsv[:, :, 1], 8, (0, 256)),
        (hsv[:, :, 2], 8, (0, 256)),
    ):
        pixels = channel[selected]
        hist = np.histogram(pixels, bins=bins, range=value_range)[0].astype(np.float32)
        values.extend(hist / max(float(hist.sum()), 1.0))
    for channel in (
        lab[:, :, 0], lab[:, :, 1], lab[:, :, 2],
        hsv[:, :, 1], hsv[:, :, 2],
    ):
        pixels = channel[selected]
        values.extend((float(pixels.mean()) / 255.0, float(pixels.std()) / 128.0))
    return np.asarray(values, dtype=np.float32)


def jersey_color_features(crops) -> np.ndarray:
    """Robust Lab/HSV histograms and moments for RGB torso crops."""
    features = []
    for crop in crops:
        features.append(_jersey_color_feature(crop))
    return np.stack(features) if features else np.empty((0, 62), dtype=np.float32)


def masked_jersey_color_features(crops, valid_masks) -> np.ndarray:
    """Jersey descriptors from guarded, crop-local foreground pixels only."""
    if len(crops) != len(valid_masks):
        raise ValueError(
            f"expected one valid mask per crop, got {len(valid_masks)} for "
            f"{len(crops)} crops"
        )
    features = [
        _jersey_color_feature(crop, valid_mask)
        for crop, valid_mask in zip(crops, valid_masks)
    ]
    return np.stack(features) if features else np.empty((0, 62), dtype=np.float32)


def record_team_vote(
    evidence: dict, team_id: int, confidence: float = 1.0, quality: float = 1.0,
) -> None:
    """Accumulate confidence- and quality-weighted evidence for one team."""
    if team_id not in (0, 1):
        return
    weight = float(np.clip(confidence, 0.0, 1.0) * np.clip(quality, 0.0, 1.0))
    if weight <= 0:
        return
    evidence[team_id] = evidence.get(team_id, 0.0) + weight


def team_vote_confidence(evidence: dict) -> float:
    """Posterior confidence of the leading team under a symmetric prior."""
    scores = np.array(
        [evidence.get(0, 0.0), evidence.get(1, 0.0)], dtype=float
    ) + TEAM_EVIDENCE_PRIOR
    return float(scores.max() / scores.sum())


def voted_team_id(evidence: dict, fallback: int) -> int:
    """Highest accumulated team evidence, or the creation-time fallback."""
    if not evidence:
        return fallback
    return max(evidence.items(), key=lambda item: item[1])[0]


class TeamModelCacheStale(Exception):
    """A cached fit predates the current crop geometry or model schema."""


class TeamModel:
    """Per-video unsupervised team model with color-first observations."""

    def __init__(
        self,
        classifier: TeamClassifier,
        color_scaler,
        color_reducer,
        color_projection_scaler,
        color_cluster_model,
        visual_to_color,
        visual_color_agreement: float,
    ):
        self._classifier = classifier
        self._color_scaler = color_scaler
        self._color_reducer = color_reducer
        self._color_projection_scaler = color_projection_scaler
        self._color_cluster_model = color_cluster_model
        self._visual_to_color = np.asarray(visual_to_color, dtype=int)
        self.visual_color_agreement = float(visual_color_agreement)

    @classmethod
    def fit_from_video(
        cls,
        video_path,
        detect_fn,
        exclude_class_ids=(1,),
        stride: int = 10,
        max_crops: int = 2000,
        seed: int = 0,
        device: str = "cuda",
    ) -> "TeamModel":
        """Fit two anonymous jersey-color prototypes from clean video crops."""
        classifier = TeamClassifier(device=device)
        crops = []
        for idx, frame_bgr in enumerate(
            sv.get_video_frames_generator(str(video_path))
        ):
            if idx % stride != 0:
                continue
            frame_rgb = frame_bgr[:, :, ::-1]
            detections = detect_fn(frame_rgb)
            if len(detections) == 0:
                continue
            contamination = torso_contamination(detections.xyxy)
            keep = ~np.isin(detections.class_id, list(exclude_class_ids))
            for box, overlap in zip(torso_boxes(detections.xyxy[keep]), contamination[keep]):
                if overlap > MAX_TORSO_CONTAMINATION:
                    continue
                crop = sv.crop_image(frame_rgb, box)
                if crop_quality(crop).accepted:
                    crops.append(crop)
            if len(crops) >= max_crops:
                break
        return cls.fit_from_crops(crops[:max_crops], seed=seed, device=device, classifier=classifier)

    @classmethod
    def fit_from_run(
        cls,
        run_npz_path,
        mask_cache_dir=None,
        exclude_goalkeepers: bool = True,
        stride: int = 5,
        max_crops: int = 2000,
        seed: int = 0,
        device: str = "cuda",
        mask_fill: str = "mean",
    ) -> "TeamModel":
        """Diagnostic fit from a saved run using the same torso rectangles.

        Masks are deliberately ignored: fitting and prediction must see the
        same pixels, and team classification must not depend on a specific
        tracker. Run row r corresponds to source frame r + 1.
        """
        del mask_cache_dir, mask_fill
        with np.load(run_npz_path, allow_pickle=True) as data:
            boxes = data["boxes"]
            source = str(data["source"])
            is_gk = (
                data["is_goalkeeper"] if "is_goalkeeper" in data.files
                else np.zeros(boxes.shape[1], dtype=bool)
            )
        crops = []
        for source_frame, frame_bgr in enumerate(
            sv.get_video_frames_generator(source)
        ):
            row_index = source_frame - 1
            if row_index < 0 or row_index >= len(boxes) or row_index % stride:
                continue
            row = boxes[row_index]
            cols = np.nonzero(np.isfinite(row).all(axis=1))[0]
            if exclude_goalkeepers:
                cols = cols[~is_gk[cols]]
            if len(cols) == 0:
                continue
            frame_rgb = frame_bgr[:, :, ::-1]
            contamination = torso_contamination(row[cols])
            for box, overlap in zip(torso_boxes(row[cols]), contamination):
                if overlap > MAX_TORSO_CONTAMINATION:
                    continue
                crop = sv.crop_image(frame_rgb, box)
                if crop_quality(crop).accepted:
                    crops.append(crop)
            if len(crops) >= max_crops:
                break
        return cls.fit_from_crops(crops[:max_crops], seed=seed, device=device)

    @classmethod
    def fit_from_crops(
        cls,
        crops,
        seed: int = 0,
        device: str = "cuda",
        classifier: TeamClassifier | None = None,
    ) -> "TeamModel":
        """Fit from RGB torso crops; useful for deterministic evaluation."""
        crops = [crop for crop in crops if crop_quality(crop).accepted]
        if len(crops) < 4:
            raise RuntimeError(
                f"only {len(crops)} usable crops sampled for team fitting"
            )
        if classifier is None:
            classifier = TeamClassifier(device=device)

        embeddings = classifier.extract_features(crops)
        classifier.reducer = umap.UMAP(
            n_components=UMAP_N_COMPONENTS, random_state=seed
        )
        visual_projection = classifier.reducer.fit_transform(embeddings)
        classifier.cluster_model = _build_cluster_model(seed)
        classifier.cluster_model.fit(visual_projection)

        color_features = jersey_color_features(crops)
        color_scaler = StandardScaler().fit(color_features)
        color_scaled = color_scaler.transform(color_features)
        color_reducer = PCA(
            n_components=min(COLOR_PCA_COMPONENTS, len(crops) - 1),
            random_state=seed,
        ).fit(color_scaled)
        color_projection = color_reducer.transform(color_scaled)
        color_projection_scaler = StandardScaler().fit(color_projection)
        color_projection = color_projection_scaler.transform(color_projection)
        color_cluster_model = KMeans(
            n_clusters=2, random_state=seed, n_init=20
        ).fit(color_projection)

        visual_ids = classifier.cluster_model.predict(visual_projection).astype(int)
        color_ids = color_cluster_model.predict(color_projection).astype(int)
        same = float(np.mean(visual_ids == color_ids))
        if same >= 0.5:
            visual_to_color = np.array([0, 1], dtype=int)
            agreement = same
        else:
            visual_to_color = np.array([1, 0], dtype=int)
            agreement = 1.0 - same

        return cls(
            classifier,
            color_scaler,
            color_reducer,
            color_projection_scaler,
            color_cluster_model,
            visual_to_color,
            agreement,
        )

    @staticmethod
    def _model_prediction(model, features):
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)
            return (
                probability.argmax(axis=1).astype(int),
                probability.max(axis=1).astype(float),
            )
        teams = model.predict(features).astype(int)
        centers = model.cluster_centers_
        distances = np.linalg.norm(
            features[:, None, :] - centers[None, :, :], axis=2
        )
        ordered = np.sort(distances, axis=1)
        denominator = ordered[:, 0] + ordered[:, 1]
        confidence = np.divide(
            ordered[:, 1] - ordered[:, 0],
            denominator,
            out=np.zeros(len(features), dtype=float),
            where=denominator > 1e-8,
        )
        return teams, confidence

    def _color_prediction(self, color_features):
        """Predict anonymous team IDs from precomputed jersey-color features."""
        features = np.asarray(color_features, dtype=np.float32)
        if len(features) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        color = self._color_scaler.transform(features)
        color = self._color_reducer.transform(color)
        color = self._color_projection_scaler.transform(color)
        return self._model_prediction(self._color_cluster_model, color)

    def _raw_crop_prediction(
        self, crops, embeddings=None, color_features=None,
    ):
        if not crops:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        if embeddings is None:
            embeddings = self.extract_features(crops)

        if color_features is None:
            color_features = jersey_color_features(crops)
        color_team, color_confidence = self._color_prediction(
            color_features
        )

        visual_team, visual_confidence = self.predict_from_embeddings(embeddings)
        use_visual = (
            (color_confidence < MIN_COLOR_CONFIDENCE)
            & (visual_confidence >= MIN_VISUAL_FALLBACK_CONFIDENCE)
            & (self.visual_color_agreement >= MIN_VISUAL_COLOR_AGREEMENT)
        )
        teams = np.where(use_visual, visual_team, color_team).astype(int)
        confidence = np.where(
            use_visual, visual_confidence, color_confidence
        ).astype(float)

        agree = visual_team == color_team
        confidence = np.where(
            agree,
            np.maximum(
                confidence,
                visual_confidence * self.visual_color_agreement,
            ),
            confidence,
        )
        return teams, np.clip(confidence, 0.0, 1.0)

    @staticmethod
    def _crop_boxes(frame_rgb: np.ndarray, boxes_xyxy: np.ndarray):
        crops = []
        for box in torso_boxes(boxes_xyxy):
            crop = sv.crop_image(frame_rgb, box)
            if crop.size == 0:
                crop = np.zeros((MIN_CROP_SIDE, MIN_CROP_SIDE, 3), dtype=np.uint8)
            crops.append(crop)
        return crops

    def observe(
        self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray,
        context_boxes_xyxy: np.ndarray | None = None,
    ):
        """Return embeddings, team ids, raw confidence, and quality weights."""
        boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
        if len(boxes) == 0:
            empty_i = np.empty(0, dtype=int)
            empty_f = np.empty(0, dtype=float)
            return np.empty((0, 0), dtype=float), empty_i, empty_f, empty_f
        crops = self._crop_boxes(frame_rgb, boxes)
        embeddings = self.extract_features(crops)
        teams, confidence = self._raw_crop_prediction(crops, embeddings)
        quality = np.array([crop_quality(crop).score for crop in crops], dtype=float)
        contamination = torso_contamination(boxes, context_boxes_xyxy)
        overlap_weight = np.clip(
            1.0 - contamination / MAX_TORSO_CONTAMINATION, 0.0, 1.0
        )
        quality *= overlap_weight
        return embeddings, teams, confidence, quality

    def predict(self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray):
        """Effective per-crop prediction; unusable crops receive confidence 0."""
        _embeddings, teams, confidence, quality = self.observe(
            frame_rgb, boxes_xyxy
        )
        return teams, confidence * quality

    def predict_crops(self, crops, embeddings=None):
        """Prediction for already-cropped RGB torso images."""
        if len(crops) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        teams, confidence = self._raw_crop_prediction(crops, embeddings)
        quality = np.array([crop_quality(crop).score for crop in crops])
        return teams, confidence * quality

    def predict_masked(
        self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray, masks: list
    ):
        """Compatibility path: team evidence intentionally ignores tracker masks."""
        del masks
        return self.predict(frame_rgb, boxes_xyxy)

    def predict_from_embeddings(self, embeddings: np.ndarray):
        """Visual-only fallback, aligned to the color cluster ids."""
        embeddings = np.asarray(embeddings)
        if len(embeddings) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        projection = self._classifier.reducer.transform(embeddings)
        teams, confidence = self._model_prediction(
            self._classifier.cluster_model, projection
        )
        return (
            self._visual_to_color[teams],
            confidence * self.visual_color_agreement,
        )

    def extract_features(self, crops):
        """Reuse the existing SigLIP pass for re-ID and visual fallback."""
        return self._classifier.extract_features(crops)

    def save(self, path) -> None:
        """Persist small per-video reducers/prototypes, not SigLIP weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "schema_version": MODEL_SCHEMA_VERSION,
                    "geometry_signature": _GEOMETRY_SIGNATURE,
                    "reducer": self._classifier.reducer,
                    "cluster_model": self._classifier.cluster_model,
                    "color_scaler": self._color_scaler,
                    "color_reducer": self._color_reducer,
                    "color_projection_scaler": self._color_projection_scaler,
                    "color_cluster_model": self._color_cluster_model,
                    "visual_to_color": self._visual_to_color,
                    "visual_color_agreement": self.visual_color_agreement,
                },
                file,
            )

    @classmethod
    def load(
        cls,
        path,
        classifier: TeamClassifier | None = None,
        device: str = "cuda",
    ) -> "TeamModel":
        with open(Path(path), "rb") as file:
            state = pickle.load(file)
        if (
            state.get("schema_version") != MODEL_SCHEMA_VERSION
            or state.get("geometry_signature") != _GEOMETRY_SIGNATURE
        ):
            raise TeamModelCacheStale(
                f"{path} was fit under a different crop geometry or model schema"
            )
        if classifier is None:
            classifier = TeamClassifier(device=device)
        classifier.reducer = state["reducer"]
        classifier.cluster_model = state["cluster_model"]
        return cls(
            classifier,
            state["color_scaler"],
            state["color_reducer"],
            state["color_projection_scaler"],
            state["color_cluster_model"],
            state["visual_to_color"],
            state["visual_color_agreement"],
        )

    @classmethod
    def load_or_fit(
        cls, cache_path, video_path, detect_fn, device: str = "cuda", **fit_kwargs
    ) -> "TeamModel":
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                return cls.load(cache_path, device=device)
            except TeamModelCacheStale as error:
                print(f"team model cache stale, refitting: {error}")
        model = cls.fit_from_video(
            video_path, detect_fn, device=device, **fit_kwargs
        )
        model.save(cache_path)
        return model
