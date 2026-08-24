# Cleanup plan

Inventory date: 2026-08-25. Batch A was completed after explicit approval;
later batches have not been deleted.

## Protected material

The following must remain until a later migration explicitly replaces them:

- `data/annotations/`: canonical human-label copies, verified byte-identical to
  their legacy originals.
- `models/`: the centralized 870 MB local checkpoint cache.
- `source/` and root-level videos: raw inputs and legacy caches until raw media
  is deliberately moved into `data/raw/` and path references are rewritten.
- modified or untracked notebooks belonging to the user.
- `SAM-MT/`: eight untracked local entries, including checkpoints and caches.
- `segment-anything-2-real-time/`: fourteen untracked local outputs/checkpoints.

## Batch A: completed

The six checkpoint copies below were verified byte-for-byte by SHA-256 and
removed after approval. They reclaimed 2,049,193,516 bytes, approximately
1.91 GiB.

| Removed duplicate copy | Preserved canonical copy |
| --- | --- |
| `notebooks/models/sam/sam_vit_b_01ec64.pth` | `models/sam/sam_vit_b_01ec64.pth` |
| `notebooks/models/cutie/cutie-base-mega.pth` | `models/cutie/cutie-base-mega.pth` |
| `segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt.1` | same path without `.1` |
| `segment-anything-2-real-time/checkpoints/sam2.1_hiera_base_plus.pt.1` | same path without `.1` |
| `segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt.1` | same path without `.1` |
| `segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt.2` | same path without `.2` |

Also removed in this batch:

- four `*.orig` files, approximately 60 KB total;
- project and vendored Python/test caches, approximately 9 MB total.

## Batch B: upstream source checkouts

These are not project source and should eventually live outside this repository.

| Path | Size | Local status | Recommendation |
| --- | ---: | --- | --- |
| `onnxruntime/` | 3.5 GB | pristine | Remove checkout after approval; runtime imports the installed package. |
| `sam2-upstream/` | 117 MB | pristine | Externalize; retained only by the SAM2 baseline. |
| `sam3/` | 133 MB | pristine | Externalize if jersey/SAM3 experiments continue. |
| `SAM-MT/` | 2.7 GB | 8 untracked entries | Do not remove wholesale. Preserve local checkpoint material first. |
| `segment-anything-2-real-time/` | 3.5 GB | 14 untracked entries | Do not remove wholesale. It contains local outputs and baseline weights. |

The ONNX Runtime checkout contains roughly 1.3 GB of build products and 1.5 GB
of Git object data. It is the best second cleanup target.

## Batch C: generated experiment results

`outputs/` occupies approximately 2.6 GB. Human labels have been copied and
verified under `data/annotations/`, but the directory also contains rendered
comparison videos that may still be useful for visual review.

| Subtree | Approximate size |
| --- | ---: |
| `outputs/team_confidence_v2/` | 880 MB |
| `outputs/team_raw/` | 674 MB |
| `outputs/team_comparison/` | 491 MB |
| `outputs/team_dataset/` | 249 MB |
| `outputs/team_correction_mcbyte/` | 219 MB |
| `outputs/mask_team_comparison/` | 78 MB |

Recommendation: archive or move selected final overlays into a small review
folder, then remove the remaining reproducible output tree only after approval.

## Legacy Python copies

The original `notebooks/*.py` files remain tracked and recoverable in Git. Once
the `src/`, `scripts/`, and `experiments/` paths have been used for normal work,
the old Python copies can be removed in a separate Git commit. They consume
little disk space; removing them is about clarity, not capacity.

## Approval sequence

1. Completed: Batch A reclaimed approximately 1.92 GiB.
2. Decide whether the ONNX Runtime checkout is still needed for development.
3. Select which rendered videos must be retained before cleaning `outputs/`.
4. Externalize remaining upstream repositories before removing legacy imports.
