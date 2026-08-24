"""Metrics for tracker-independent, anonymous two-team discovery."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score


@dataclass(frozen=True)
class ClusterPrediction:
    labels: np.ndarray
    confidence: np.ndarray
    fit_count: int


def l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"expected a feature matrix, got shape {features.shape}")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(
        features,
        norms,
        out=np.zeros_like(features),
        where=norms > 1e-12,
    )


def spherical_two_means(
    features: np.ndarray,
    fit_mask: np.ndarray | None = None,
    seed: int = 0,
) -> ClusterPrediction:
    """Fit two clusters on normalized embeddings and predict every sample."""
    normalized = l2_normalize(features)
    if fit_mask is None:
        fit_mask = np.ones(len(normalized), dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if fit_mask.shape != (len(normalized),):
        raise ValueError("fit_mask must contain one boolean per feature")
    if int(fit_mask.sum()) < 2:
        raise ValueError("at least two fit samples are required")
    model = KMeans(n_clusters=2, random_state=seed, n_init=20).fit(
        normalized[fit_mask]
    )
    distances = model.transform(normalized)
    labels = distances.argmin(axis=1)
    own = distances[np.arange(len(distances)), labels]
    other = distances[np.arange(len(distances)), 1 - labels]
    confidence = np.divide(
        other - own,
        other + own,
        out=np.zeros(len(distances), dtype=float),
        where=(other + own) > 1e-12,
    )
    return ClusterPrediction(
        labels=labels.astype(int),
        confidence=np.clip(confidence, 0.0, 1.0),
        fit_count=int(fit_mask.sum()),
    )


def align_anonymous_binary_labels(
    cluster_labels: np.ndarray, ground_truth: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Choose the A/B permutation only for scoring anonymous clusters."""
    predicted = np.asarray(cluster_labels, dtype=int)
    truth = np.asarray(ground_truth, dtype=int)
    if predicted.shape != truth.shape:
        raise ValueError("prediction and ground truth shapes differ")
    if not set(np.unique(predicted)).issubset({0, 1}):
        raise ValueError("cluster labels must be binary")
    if not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("ground-truth labels must be binary")
    direct = float(np.mean(predicted == truth)) if len(truth) else 0.0
    swapped = float(np.mean((1 - predicted) == truth)) if len(truth) else 0.0
    if swapped > direct:
        return 1 - predicted, {0: 1, 1: 0}
    return predicted.copy(), {0: 0, 1: 1}


def accuracy_at_coverages(
    correct: np.ndarray,
    confidence: np.ndarray,
    coverages: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5),
) -> dict[str, dict[str, float | int]]:
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    if correct.shape != confidence.shape:
        raise ValueError("correctness and confidence shapes differ")
    if not len(correct):
        return {}
    order = np.argsort(-confidence, kind="stable")
    result = {}
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError(f"invalid coverage: {coverage}")
        count = max(1, int(np.ceil(len(correct) * coverage)))
        selected = order[:count]
        result[f"{coverage:.2f}"] = {
            "count": count,
            "accuracy": float(correct[selected].mean()),
            "minimum_confidence": float(confidence[selected].min()),
        }
    return result


def score_anonymous_teams(
    cluster_labels: np.ndarray,
    confidence: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """Score binary team clusters after the necessary permutation alignment."""
    aligned, mapping = align_anonymous_binary_labels(cluster_labels, ground_truth)
    truth = np.asarray(ground_truth, dtype=int)
    confidence = np.asarray(confidence, dtype=float)
    correct = aligned == truth
    return {
        "count": int(len(truth)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, aligned)),
        "macro_f1": float(f1_score(truth, aligned, average="macro")),
        "ari": float(adjusted_rand_score(truth, cluster_labels)),
        "cluster_to_team": {str(key): int(value) for key, value in mapping.items()},
        "mean_confidence": float(confidence.mean()),
        "coverage": accuracy_at_coverages(correct, confidence),
    }


def evaluate_manifest_embeddings(
    manifest: dict,
    features: np.ndarray,
    max_fit_overlap: float = 0.25,
    seed: int = 0,
) -> dict:
    """Fit without labels, then score only manually labeled field players."""
    samples = manifest["samples"]
    features = np.asarray(features)
    if len(features) != len(samples):
        raise ValueError("feature count does not match manifest samples")
    fit_mask = np.array([
        sample.get("detector_class_id") == 2
        and bool((sample.get("crop_quality") or {}).get("accepted", False))
        and float(sample.get("torso_contamination", 1.0)) <= max_fit_overlap
        for sample in samples
    ])
    prediction = spherical_two_means(features, fit_mask=fit_mask, seed=seed)
    evaluation_indices = np.array([
        index for index, sample in enumerate(samples)
        if sample.get("annotation", {}).get("team") in {"A", "B"}
        and sample.get("annotation", {}).get("role") == "field"
        and sample.get("annotation", {}).get("quality") not in {"mixed", "unusable"}
    ], dtype=int)
    if not len(evaluation_indices):
        raise ValueError(f"{manifest.get('video_id')} has no labeled A/B field players")
    truth = np.array([
        0 if samples[index]["annotation"]["team"] == "A" else 1
        for index in evaluation_indices
    ])
    result = score_anonymous_teams(
        prediction.labels[evaluation_indices],
        prediction.confidence[evaluation_indices],
        truth,
    )
    result.update({
        "video_id": manifest.get("video_id"),
        "fit_count": prediction.fit_count,
        "total_samples": len(samples),
    })
    for subset_name, subset_indices in {
        "clean": np.array([
            index for index in evaluation_indices
            if samples[index]["annotation"]["quality"] == "clean"
        ], dtype=int),
        "occluded": np.array([
            index for index in evaluation_indices
            if samples[index]["annotation"]["quality"] == "occluded"
        ], dtype=int),
    }.items():
        if not len(subset_indices):
            continue
        local_truth = np.array([
            0 if samples[index]["annotation"]["team"] == "A" else 1
            for index in subset_indices
        ])
        # Apply the video-level semantic mapping, not a separately optimized
        # mapping that could make a hard subset look artificially better.
        mapping = {int(k): v for k, v in result["cluster_to_team"].items()}
        aligned = np.array([mapping[int(value)] for value in prediction.labels[subset_indices]])
        result.setdefault("subsets", {})[subset_name] = {
            "count": int(len(subset_indices)),
            "accuracy": float(np.mean(aligned == local_truth)),
        }
    return result
