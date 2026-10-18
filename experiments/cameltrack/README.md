# CAMELTrack GPU comparison

Prepared in branch `experiment/cameltrack`, worktree
`/home/valentinweyer/projects/handball-computer-vision-camel`.
Production tracking still uses SAM2. No real clip was processed during setup.

The setup uses the official [CAMELTrack](https://github.com/TrackingLaboratory/CAMELTrack)
SportsMOT checkpoint with its learned appearance, keypoint and bounding-box cues:

- Native CAMEL association and track lifecycle, FP32 on CUDA.
- Released [KPR](https://github.com/victorjoos/keypoint_promptable_reidentification)
  Swin-base checkpoint, FP32 on CUDA. Native evaluation transforms and feature
  extraction produce six 128-dimensional embeddings with visibility scores.
- Official RTMPose-m 256x192 ONNX model through RTMLib and CUDA ONNX Runtime.
- Existing cached RF-DETR goalkeeper/player detections (classes 1 and 2).

Both PyTorch checkpoints load strictly, with no missing or unexpected tensors.
No PRTReID substitution, fabricated features, training or local threshold tuning.
The SportsMOT settings are detection confidence 0.1, initialization 0.4,
minimum hits 0, lost lifetime 150 frames, gallery size 50.
Sources, model download revisions and SHA-256 hashes are in `sources.lock.json`.

## Setup validation

The dedicated `.venv` has its own optional dependencies. Its
`handball_shared_cuda.pth` reads the original checkout's installed packages,
including the existing `torch==2.9.1+cu130`; it does not install into that environment.
The overlay includes GPU ONNX Runtime 1.31.0, because the base runtime is CPU-only.
Use this worktree's `.venv/bin/python` explicitly. Do not use `uv run` or sync the
main project's dependencies into this experimental environment.

On the NVIDIA GB10, the synthetic GPU preflight passed strict model loading,
three association updates, an empty frame, reappearance and reset. ONNX profiling
recorded 1,850 CUDA node executions, including all pose convolutions; 20 small
shape-related node executions used CPU. Image decoding, crop transforms and
Hungarian assignment also use CPU, as in the native pipeline.
Evidence: `.local/cameltrack/setup_gpu.json` and `preflight-gpu.log`.
This validates GPU operation, not speed or player-tracking accuracy.

```bash
cd /home/valentinweyer/projects/handball-computer-vision-camel
# CPU-only synthetic validation; hides CUDA before importing the models.
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.cameltrack.preflight \
  --device cpu --output .local/cameltrack/setup_cpu.json

# Explicit GPU synthetic validation. Does not open a clip.
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.cameltrack.preflight \
  --device cuda --output .local/cameltrack/setup_gpu.json
```

## Next step: run a clip (not executed during setup)

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.cameltrack.run_comparison \
  --case felix --max-frames 141 --output runs/cameltrack/felix141

.venv/bin/python -m experiments.cameltrack.score runs/cameltrack/felix141 \
  --baseline /home/valentinweyer/projects/handball-computer-vision-sam31/runs/sam31_full_20261015/felix_sam2_baseline
```

Cases: `felix`, `han`, `bhc`. Omit `--max-frames` for the full cached clip.
Each run requires a new output directory. It writes `replay.npz`, model/input
fingerprints, frame counts and measured stage timings in `manifest.json`.
The scorer produces `identity_score.json` on CPU using the existing identity
evaluator, trusted human verification spans and IoU threshold 0.5. Baseline
input hashes must match; both outputs are scored over the candidate's exact
frame prefix. Coverage and correct-identity percentage use **all trusted player
observations** as the denominator, so dropped players cannot improve the score.
Identity switches and fragmentation are also reported. This local metric is
not public HOTA or IDF1, and is not independent ground truth for SAM2.

Loop FPS includes JPEG decoding, cached-detection filtering, pose, appearance,
association and lifecycle. Model loading, synthetic warmup and artifact writing
are separate. There are no live RF-DETR, OCR, team-classification or mask costs.
CAMEL uses its native lifecycle; SAM2's historical replay includes its existing
reprompt/recovery policy, so this compares complete tracking paths rather than
isolating the association architecture. Compare quality on the same prefix;
historical full-clip FPS has a different timing span and setup overhead.
First clip results are recorded in [RESULTS.md](RESULTS.md): 5.47 FPS, with a substantial identity-quality reduction versus SAM2.

## Recreating this machine-local environment

The overlay requirements assume this machine's working base environment; they
are not a portable CUDA installation recipe. Model assets live under ignored
`.local/cameltrack/`. Dataset links point read-only consumers at the original
checkout. The runtime inserts only the three pinned source checkouts and this
worktree's `src` on the import path.

```bash
# Run in a fresh experiment worktree. Setup commands hide the GPU.
export CUDA_VISIBLE_DEVICES=''
BASE=/home/valentinweyer/projects/handball-computer-vision
"$BASE/.venv/bin/python" -m venv .venv
.venv/bin/python - "$BASE" <<'PY'
from pathlib import Path
import sys, sysconfig
base = Path(sys.argv[1]) / '.venv/lib/python3.11/site-packages'
(Path(sysconfig.get_paths()['purelib']) / 'handball_shared_cuda.pth').write_text(str(base) + '\n')
PY
.venv/bin/python -m pip install --no-deps -r experiments/cameltrack/requirements.txt

# Fetch the exact official source commits in the lock file (no model execution).
.venv/bin/python - <<'PY'
import json, subprocess
from pathlib import Path
assets = Path('.local/cameltrack')
assets.mkdir(parents=True, exist_ok=True)
lock = json.loads(Path('experiments/cameltrack/sources.lock.json').read_text())
for name, source in lock['sources'].items():
    target = assets / name
    subprocess.run(['git', 'init', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'fetch', '--depth=1', source['url'], source['revision']], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
PY
hf download trackinglaboratory/CAMELTrack camel_bbox_app_kps_sportsmot.ckpt \
  --revision 6759b7c27b99d8f2b8d8a13be69c6dbbb9cc40ff --local-dir .local/cameltrack/checkpoints
hf download trackinglaboratory/keypoint_promptable_reid \
  kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar \
  --revision 8b062e905989609060c700e7eb526950e220628b --local-dir .local/cameltrack/checkpoints
# Download the OpenMMLab ZIP URL in sources.lock.json and extract its .onnx
# as .local/cameltrack/checkpoints/rtmpose-m.onnx; verify the listed SHA-256.
# Link existing input directories if this is a fresh worktree:
mkdir -p data runs
ln -s "$BASE/data/cache" data/cache
ln -s "$BASE/data/raw" data/raw
ln -s "$BASE/outputs" outputs
ln -s "$BASE/source" source
ln -s "$BASE/runs/tracklet_labels" runs/tracklet_labels
```

For a GPU run in a new shell, `CUDA_VISIBLE_DEVICES` must permit GPU 0. The
runner requires CUDA for all three neural components and rejects a CPU pose
provider fallback. It does not modify the original checkout or its environment.
