# Outputs migration reference report

Inventory date: 2026-08-25.

> Planning artifact only: no file under `outputs/`, and no existing code, notebook, metadata, or annotation file, was moved, renamed, edited, or deleted.

[`outputs-migration-manifest.csv`](outputs-migration-manifest.csv) inventories 3,313 files totaling 2.53 GiB with proposed destinations, reference locations, encode pairs, and SHA-256 checksums.

## Critical dependency discovered

The manifests under `outputs/team_dataset/<video>/manifest.json` are byte-identical to the flat copies under `data/annotations/team/`, but their `crops/...`, `torsos/...`, and `previews/...` paths are relative to the manifest directory.

| Video | Samples | Missing assets beside flat canonical copy | Missing assets beside current output manifest |
| --- | ---: | ---: | ---: |
| FelixClaar | 206 | 618 | 0 |
| Han-Ber4 | 700 | 2,100 | 0 |
| Hannover | 144 | 432 | 0 |

The three `outputs/team_dataset/<video>/` bundles are active dependencies. They must remain intact until manifest and assets migrate together. `outputs/team_dataset/Han-Ber4_cached.mp4` is also an active source clip; its proposed destination is `data/raw/Han-Ber4.mp4`.

## Reference audit method

The audit scanned tracked and visible untracked text outside `outputs/`, notebook cell sources (excluding embedded outputs), text metadata inside `outputs/`, relative dataset links, and dynamically constructed `outputs/<subtree>` defaults. Binary files are inventoried and hashed; arbitrary serialized object graphs were not executed as reference sources.

Notebook reference locations use the ordinal line within extracted cell source, not the raw JSON file line.

## Reference status

| Status | Files |
| --- | ---: |
| `direct_external_reference` | 24 |
| `directory_level_reference` | 72 |
| `internal_output_dependency` | 34 |
| `no_text_reference_found` | 28 |
| `relative_external_dependency` | 3,155 |

## Proposed categories

| Category | Files | Size |
| --- | ---: | ---: |
| `superseded_confidence_run` | 13 | 874.83 MiB |
| `raw_frame_team_baseline` | 10 | 673.43 MiB |
| `superseded_team_comparison` | 19 | 481.92 MiB |
| `current_temporal_team_run` | 25 | 217.55 MiB |
| `raw_evaluation_clip` | 1 | 159.50 MiB |
| `mask_overlap_evaluation` | 12 | 77.02 MiB |
| `annotation_review_asset` | 3,150 | 74.91 MiB |
| `per_video_team_model` | 8 | 9.50 MiB |
| `early_labeling_trial` | 26 | 7.25 MiB |
| `tracklet_diagnostic` | 5 | 4.81 MiB |
| `embedding_cache` | 8 | 3.96 MiB |
| `legacy_label_sheet` | 2 | 1.32 MiB |
| `human_annotation_manifest` | 3 | 1.13 MiB |
| `calibration_review_artifact` | 7 | 1008.74 KiB |
| `detection_cache` | 4 | 906.00 KiB |
| `labeling_interface` | 3 | 865.44 KiB |
| `embedding_evaluation` | 12 | 13.33 KiB |
| `dataset_audit` | 1 | 10.14 KiB |
| `legacy_manual_label_artifact` | 4 | 7.30 KiB |

## Proposed actions

| Action | Files | Size |
| --- | ---: | ---: |
| `review_redundant_encode` | 17 | 1.48 GiB |
| `migrate_run_artifact` | 61 | 563.62 MiB |
| `archive` | 53 | 258.12 MiB |
| `migrate_and_repoint` | 20 | 173.59 MiB |
| `bundle_with_manifest` | 3,154 | 75.76 MiB |
| `bundle_assets_then_repoint` | 3 | 1.13 MiB |
| `verify_then_migrate` | 1 | 256.84 KiB |
| `verify_against_canonical_then_archive` | 4 | 7.30 KiB |

## Directory-level references

- `outputs/mask_team_comparison/`: `docs/cleanup-plan.md:68`
- `outputs/team_comparison/`: `docs/cleanup-plan.md:65`, `experiments/team_gated_tracking/render_comparison.py:62`, `notebooks/calibrate_teams.py:30`, `notebooks/label_team_detections.py:39`, `notebooks/render_team_comparison.py:62`, `notebooks/render_tracklet_sheets.py:38`, `scripts/calibrate_teams.py:30`, `scripts/label_team_detections.py:39`, `scripts/render_tracklet_sheets.py:38`
- `outputs/team_comparison/calibration/`: `experiments/team_gated_tracking/render_comparison.py:74`, `notebooks/render_team_comparison.py:74`
- `outputs/team_comparison/tracklets/`: `notebooks/render_tracklet_sheets.py:42`, `scripts/render_tracklet_sheets.py:42`
- `outputs/team_confidence_v2/`: `docs/cleanup-plan.md:63`
- `outputs/team_correction_mcbyte/`: `docs/cleanup-plan.md:67`
- `outputs/team_dataset/`: `data/README.md:10`, `docs/cleanup-plan.md:66`, `notebooks/label_team_detections.py:40`, `scripts/label_team_detections.py:40`
- `outputs/team_raw/`: `docs/cleanup-plan.md:64`

## Interpretation

- Active-input actions are not deletion candidates.
- `archive` means superseded but potentially useful.
- `review_redundant_encode` marks a possible MP4/H.264 pair; it does not assert equivalence.
- `no_text_reference_found` does not prove a file is unused; CLI and human workflows may consume it.

## Before any future migration

1. Approve destinations by category.
2. Copy one category at a time and verify SHA-256.
3. Rewrite exact, relative, and directory references.
4. Recheck manifest asset resolution and run tests.
5. Compare paired videos before selecting an encode.
6. Delete only after separate explicit approval.
