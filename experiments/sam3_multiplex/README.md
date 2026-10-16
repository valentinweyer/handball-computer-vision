# SAM3.1 multiplex experiment

Isolated, non-default GPU experiment using the existing RF-DETR caches and
production `drive_sam2` / `TrackManager` correction policy. **The short-run
deficit is concentrated in one target; full-clip comparisons are mixed so far.
Quality equivalence remains inconclusive, and BHC exposes a multi-bucket
temporal-memory crash that blocks adoption.** See
[the measured results](../../docs/sam31-multiplex-results.md).

No packages or weights were downloaded. Existing source: `sam3/` at
`5a3143c652ac41c28b788a419b40ec424d0f6aab`. Existing checkpoint:
`sam3-checkpoints/sam3.1_multiplex.pt`, SHA256
`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
Use the existing `.venv`; do not replace its aarch64 CUDA PyTorch packages.

## Adapter details

- Strictly loads all 931 tracker/shared-encoder tensors (486,715,574 parameters).
  Language and detector heads are omitted; the shared vision encoder is retained.
- Runs on CUDA with BF16 autocast, native 1008 input, eager execution. SAM2 uses
  its native 1024 input under the same precision and manager.
- Converts RF-DETR boxes to masks with SAM3's trained interactive head, using
  box-corner labels 2/3. Masks are inserted in a batch through the official mask
  API; corrections recondition existing targets. This is an adapter configuration,
  not a claim of reproducing Meta's benchmark protocol.
- Requests only interactive and propagation encoder necks. The upstream standalone
  demo's unused detection-neck request otherwise raises an `IndexError` in
  `forward_image`. The fix is local to the experiment; upstream files are unchanged.
- Uses a one-frame lazy CPU image cache with upstream PIL resize and FP16
  normalization; the model and temporal state stay on GPU. The current frame is
  transferred inside the measured loop. This prevents full-video preloading.
- Reuses production mask edge filtering and mask-derived boxes.
- Does not reset the complete memory bank at checkpoints. Late object insertion
  is implemented through upstream APIs. A later GPU lifecycle probe in the new
  worktree exercises late insertion, correction, middle removal and per-player
  reset while checking survivor prediction history.

## Reproduce explicitly

Use a new output directory for every invocation; existing results are protected.
Run one GPU arm at a time, with sufficient unified-memory headroom. In the
worktree, export `PYTHONPATH="$PWD/src:$PWD"` first so the shared environment
imports this checkout rather than the original editable installation.

```bash
.venv/bin/python -m experiments.sam3_multiplex.run_comparison \
  --backend sam3 --case felix --max-frames 141 --warmup-frames 12 \
  --capture-masks --output runs/sam31_quality_NEW/sam3_felix141
.venv/bin/python -m experiments.sam3_multiplex.score \
  --run runs/sam31_quality_NEW/sam3_felix141

.venv/bin/python -m experiments.sam3_multiplex.run_comparison \
  --backend sam2 --case felix --max-frames 141 --warmup-frames 12 \
  --capture-masks --output runs/sam31_quality_NEW/sam2_felix141
.venv/bin/python -m experiments.sam3_multiplex.score \
  --run runs/sam31_quality_NEW/sam2_felix141
```

The scorer excludes all unprocessed reference frames from the denominator and
preserves human-verified reference lifespan limits. Its overlap-only dominant
identity mapping is local to that window; use the prefix result and visual
review together. No new quality tolerance is introduced.

Timing includes the shared tracking loop, periodic corrections, CPU masks/boxes,
SAM3 frame preparation, and diagnostic mask capture. It excludes model building,
warmup, team-model loading, live RF-DETR, OCR and rendering. This is a single
diagnostic repeat, not a deployment-speed benchmark. The seed call buffers SAM3
boxes until the first propagation; that initialization cost is in its loop time.

Artifacts: replay, per-frame timings/bucket counts, lifecycle events, manifest
with hashes, score/matches JSON and masks for frames 90–140. The tested runs and
paired visual sheets live under `runs/sam31_quality_20261015/`.

## Worktree and lifecycle follow-up

All work after the user's isolation request lives on branch
`experiment/sam31-multiplex` in
`/home/valentinweyer/projects/handball-computer-vision-sam31`.
Weights, source checkouts, datasets and the existing environment are shared
inputs; new result directories belong to this worktree. Use
`PYTHONPATH="$PWD/src:$PWD"` when invoking the shared environment so editable
imports resolve to this worktree.

The first lifecycle probe exposed an upstream edge case: a late-added object's
first prompt could be stored as non-conditioning, allowing removal of the last
originally prompted object to reset all tracking history. The adapter now marks
new-target observations as conditioning anchors. Existing-target corrections
retain their previous policy. No upstream source was edited.

The repeated real-GPU probe passed all four checks, with one multiplex bucket,
finite CUDA masks, correct public IDs and retained survivor historical mask
predictions. Removing/re-adding one ID also removes its old per-object prediction
history. This verifies API/state behavior on the probe; it does not prove
identity quality on entire matches or byte-identical shared memory.

```bash
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m \
  experiments.sam3_multiplex.audit_lifecycle \
  --output runs/sam31_lifecycle_NEW
```

Successful report: `runs/sam31_lifecycle_20261015_fixed/report.json`.
The first failed probe's log is retained separately at
`runs/sam31_lifecycle_20261015/run.log`.

The user considers the measured 2-to-3.3 FPS gain worthwhile. 30 FPS is a
longer-term goal, not the threshold for pursuing this integration. Accuracy
remains a separate requirement; the first short clip did not establish a
general regression or equivalence.

## Full-clip status

Felix and Han completed with matched historical SAM2 inputs; scores are in
`runs/sam31_full_20261015/`. BHC failed after frame 370 with incompatible
object-pointer bucket dimensions. Its log and partial mask artifacts are retained;
there is no completed BHC replay or quality score. The one-bucket lifecycle
probe does not validate bucket growth. A separate short GPU probe reproduced
the failure at 16-to-17 IDs, with older pointers retaining one bucket while the
new conditioning frame contains two. Its report is
`runs/sam31_bucket_growth_20261015/report.json`. See the results document for
counts and performance-comparison limits.

## Watch saved-result previews

`runs/sam31_previews_20261015/` contains full Felix and Han side-by-side box/ID
comparisons (SAM2 left, SAM3.1 right), a paired Felix mask excerpt, and a
SAM3.1-only BHC mask excerpt. Full clips play at source speed; mask excerpts
cover frames 90–140 at half speed. BHC is explicitly labelled as an incomplete
benchmark and does not show its later failure at frame 370. Playback speed is
independent of inference throughput. Rendering uses saved JPEGs and predictions,
without model loading or GPU inference.

```bash
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m \
  experiments.sam3_multiplex.render_previews --output runs/sam31_previews_NEW
```
