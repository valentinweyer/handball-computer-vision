# Standalone Cutie comparison

Branch `experiment/cutie`, separate worktree:
`/home/valentinweyer/projects/handball-computer-vision-cutie`.
Production SAM2 and the user's SAM3.1 worktree are unchanged.

[First results](RESULTS.md): Cutie-480 achieved 8.25 FPS and 98.91% correct
identity on the 141-frame Felix prefix; paired SAM2 achieved 2.51 FPS and 96.84%.
The reference caveat below requires confirmation on another clip.

The experiment replaces temporal propagation with the official Cutie-base-mega
model. SAM2.1-L's image head converts RF-DETR boxes into initial/correction
masks. SAM2 does not propagate those masks. The unchanged production
`TrackManager`, `PlayerRegistry`, detector checkpoint cadence (10 frames),
post-processing, and historical team-model re-ID configuration remain shared
with the SAM2 tracking-only comparison. No training or threshold tuning.

Cutie defaults to a 1024-pixel maximum short edge. `--cutie-size 480` selects
the authors' common scripting resolution for a separately measured comparison.
Both use BF16 autocast, native generic-video
long-term memory settings and memory updates every five Cutie steps. The box
prompt head uses the existing SAM2.1-L checkpoint with its native 1024 image
encoder. Existing per-video team model/cache hashes are recorded. There is no
live detection, OCR or rendered mask capture in the timed tracking loop.

## Full validation

[Full-clip validation results](VALIDATION.md) cover Han (199 frames), BHC (500),
and Felix (249), each paired with a fresh SAM2 run. Cutie-480 was 3.09–3.25x
faster, with higher correct-identity scores on Felix/BHC and a 1.60-point
coverage deficit on Han. Han had no measured ID switches or fragmentation for
either tracker. A no-loss replacement across all clips is not established.

Use `--case felix|han|bhc` with `--max-frames` to select the prefix. The scorer
uses the matching verified reference and excludes Han bench/referee IDs 12/14.

## GPU validation and lifecycle compatibility

The isolated `.venv` reads the original environment's packages through
`handball_shared_cuda.pth`; no original packages were installed or changed.
The adapter prepends the official Cutie sources and the existing official SAM2
checkout, while preserving the repository's own `scripts` import precedence.
Both checkpoints load strictly: 527 Cutie tensors and 903 SAM2 tensors, with
no missing/unexpected keys. The cached Cutie weights match the official release
MD5 `a6071de6136982e396851903ab4c083a`.

The GPU synthetic preflight passes propagation after middle-ID deletion, late
addition, same-ID remove/re-add, removal of all objects, reset and batched box
prompt conversion. It checks surviving mask channels and object memory directly.
Evidence is `.local/cutie/setup_cuda.json` and `preflight-gpu.log`.

Native lifecycle details need compatibility handling:

- The adapter selects surviving `last_mask` channels after IDs are renumbered.
- It discards removed object summaries, so a same-ID reset starts fresh.
- `upstream-delete.patch` also corrects completeness detection for partial mask
  updates: incoming masks must cover every current object before propagation
  can be skipped. A correction of one existing player must preserve the others.
- The same patch makes deletion of absent optional selection/usage
  arrays safe for permanent-only memory buckets. This fixes a native `KeyError`
  found by the synthetic test. It changes no surviving memory or learned model.

Prompts at one checkpoint are batched. A repeated boundary with no correction
uses its cached output. A corrected boundary receives one native Cutie step on
that same image, retaining the other objects' memory. This extra step advances
Cutie's native memory cadence; model timing and step counts include it. The
shared SAM2 driver otherwise remains unchanged.

## Commands

```bash
cd /home/valentinweyer/projects/handball-computer-vision-cutie
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.cutie.preflight --device cuda

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.cutie.run_comparison \
  --backend cutie --max-frames 141 --warmup-frames 12 --output runs/cutie/felix141_1024

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.cutie.run_comparison \
  --backend cutie --cutie-size 480 --max-frames 141 --warmup-frames 12 --output runs/cutie/felix141_480

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.cutie.run_comparison \
  --backend sam2 --max-frames 141 --warmup-frames 12 --output runs/cutie/sam2_felix141

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.cutie.score \
  --cutie runs/cutie/felix141_480 --sam2 runs/cutie/sam2_felix141 \
  --camel /home/valentinweyer/projects/handball-computer-vision-camel/runs/cameltrack/felix141
```

Use a new output directory per run. Each manifest records source/patch/input
hashes, resolved Cutie settings, GPU, precision and measured scope. Model build,
warmup and artifact writing are excluded. Seed and loop timing are reported
separately; **total tracking FPS** includes initialization and all 141 frames.
Cutie's queued mask generation is included in its loop, whereas SAM2's eager
seed work is in seed time, so compare total tracking FPS as well as loop FPS.
CAMEL's earlier timing used a different warmup and lifecycle; label it separately.

The scorer uses existing IoU>=0.5 matching, exact matching frame prefixes and
trusted human verification spans. Missing player observations count against
correct-identity percentage. **The Felix reference was generated with SAM+Cutie**,
so it is not an independent reference for Cutie. A successful result would need
confirmation on other clips and visual inspection before adoption.

## Recreating the local setup

Source revisions and checkpoint hashes are in `sources.lock.json`. Clone Cutie
at the recorded revision into `.local/cutie/upstream`, then apply the recorded
patch with `git -C .local/cutie/upstream apply` and its absolute path. Cached
weights are read from the original checkout; official Cutie weights are at
https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth.
The SAM2 source and checkpoint paths are explicit in `adapter.py`.

Create a venv using the original checkout's Python 3.11, then put its existing
site-packages path in the new environment's `handball_shared_cuda.pth`. This
machine-local overlay reuses CUDA PyTorch 2.9.1+cu130. Link `data/cache`, `data/raw`,
`outputs`, `source` and `runs/tracklet_labels` read-only consumers to the original
checkout. Do not run dependency sync against this experimental environment.
