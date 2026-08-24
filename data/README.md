# Data contract

This directory separates irreplaceable annotations from reproducible artifacts.

- `annotations/` contains small human-authored labels and is tracked.
- `raw/` contains source videos and images and is ignored.
- `interim/` contains generated crops or extracted frames and is ignored.
- `cache/` contains detections, masks, embeddings, and fitted per-video models and is ignored.

The current manifests were copied from `outputs/team_dataset/`; their originals
remain untouched during the non-destructive migration. Some legacy manifests
still reference their original absolute video paths. A later data migration can
rewrite those paths after the raw videos are deliberately relocated.

Canonical team ground truth is tracker-independent and keyed to RF-DETR
detections. Tracklet labels are retained separately because they are useful for
diagnostics but do not define the classification target.

Do not put API keys, model weights, rendered videos, or third-party repositories
under this directory.
