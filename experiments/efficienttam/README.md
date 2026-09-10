# EfficientTAM comparison

Experimental only. SAM2 remains the default. The sole production change is an
optional predictor factory in `drive_sam2`; the manager, registry, checkpoint
policy and frame results stay shared.

Use the source revisions recorded in
[the lifecycle investigation](../../docs/efficienttam-lifecycle-investigation.md).
No upstream package installation is necessary. Each backend must run in a fresh
process to avoid conflicting Hydra configuration roots. Fetch only the official
Small checkpoint, not the entire collection:

```bash
hf download yunyangx/efficient-track-anything efficienttam_s.pt \
  --revision 9bdd8ab --local-dir models/efficienttam
```

Trained lifecycle check:

```bash
PYTHONPATH=/path/to/EfficientTAM uv run --no-sync python \
  -m experiments.efficienttam.audit_lifecycle \
  --frames data/cache/frames/Han-Ber4_cached \
  --checkpoint models/efficienttam/efficienttam_s.pt
```

One paired case (also use `--case felix` and `--case bhc`):

```bash
uv run --no-sync python -m experiments.efficienttam.run_comparison \
  --backend sam2 --case han --upstream sam2-upstream \
  --checkpoint segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt \
  --output runs/efficienttam_pair/han_sam2_r1
uv run --no-sync python -m experiments.efficienttam.run_comparison \
  --backend efficienttam --case han --upstream /path/to/EfficientTAM \
  --checkpoint models/efficienttam/efficienttam_s.pt \
  --output runs/efficienttam_pair/han_efficienttam_r1
uv run --no-sync python -m experiments.efficienttam.score_comparison \
  --run runs/efficienttam_pair/han_sam2_r1
uv run --no-sync python -m experiments.efficienttam.score_comparison \
  --run runs/efficienttam_pair/han_efficienttam_r1
```

Outputs must be new directories. Runs retain checkpoint/input/source hashes,
model configuration, effective-operation warnings, setup and complete-loop
measurements, active-object counts, boxes, lifecycle events, per-frame matches,
and per-reference timelines. The complete-loop timer exhausts the generator,
including work after yields and at checkpoint boundaries. Compressed artifact
writing is outside timing. Warmup uses the first 21 JPEGs in its own physical
cache and its own state/manager; measured initialization uses a fresh state.

The workload is the existing **tracking-only** configuration: cached RF-DETR,
fitted TeamModel and its appearance features, every player/goalkeeper active,
interval 10, 1024, BF16 and unchanged mask post-processing. Both models are
explicitly eager. PRTReID/PARSeq and rendering are not in this historical
configuration; this is not a full-application throughput claim.

`--capture-masks` retains packed masks for every output frame in a separate
diagnostic run. It adds CPU work and memory, so do not use its timing as the
benchmark. Unpack with `np.unpackbits(data['masks'], axis=-1)[..., :int(data['width'])]`.
Check its replay and events against the measured arm before using its masks to
explain that arm's result.

The scorer deliberately reuses the historical greedy IoU matching. Its additional
constant-ID runs and misses are **reference-box match diagnostics**, not proof
of a true body swap or identity loss. Mask-shape changes and heavy overlap can
move a box across IoU 0.5. Same-team flags use existing verified Han-Ber4 labels;
other clips retain unknown flags pending visual inspection. Full occlusion,
later entrants and reference truncations still need explicit event review.

`historical_correct_pct` keeps the original all-reference denominator.
`correct_pct` uses a denominator restricted to the same player IDs as the
numerator. They differ on Han-Ber4, where bench/referee references are excluded
from the player metrics. Counts and both denominators are retained.
