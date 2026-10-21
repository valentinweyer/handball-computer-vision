# ARC-SORT CPU accuracy screen, 2026-10-21

This isolated experiment scores ARC-SORT on the existing FelixClaar,
Han-Ber4 and BHC-FAG trusted spans. It does not integrate a tracker, change
production defaults, benchmark speed, or use the GPU. Source lives on
`experiment/arc-sort-cpu`, based on `161dcf1` so it can reuse the latest
oracle-study harness. The main checkout and the full inference job stay intact.

## Inputs and methods

- Vendor: Athlitix/arcsort at `2a580c592c838bce99b89a6a2dbd7ad6f50c84be`.
- Detector: the existing `detector-expanded-v1` caches produced on **every
  frame**, from `runs/tracker_eval/oracle_2026-10-20/detections/`. Require one
  offset slot per decoded video frame and a populated slot on every frame
  of these three clips. No stride-5 input or interpolated detection substitute.
- Track field players and goalkeepers (classes 2 and 1), excluding number,
  ball and referee detections from every candidate and box baseline.
- Important limitation: these production caches have confidence floor 0.3,
  so the 0.1–0.3 portion of ARC-SORT's low-confidence recovery is absent.
  This screen does not measure its full potential with lower-floor detections.
- Appearance: the official Deep-EIoU `sports_model.pth.tar-60` OSNet release,
  Google Drive file `14zzlm1nI9Ws_Il9RYNChwPC7Fsul7xwl`, linked from the
  author's README. Weight SHA-256 is recorded in the protocol and manifest.
  Strict state-dict loading rejects missing/unused network tensors.
- OSNet uses the upstream crop rounding/clipping, BGR→RGB, 128×256 resize,
  ImageNet normalization and L2-normalized features, computed on CPU in FP32.
  Upstream live ReID hardcodes CUDA/FP16, so the experiment supplies its
  documented embedding cache instead; any cache miss raises rather than
  falling back to GPU or silently zeroing features. CPU FP32 can differ from
  the published GPU FP16 results near association thresholds.
- Run the SportsMOT preset unchanged, with and without appearance. A secondary
  configuration uses high/new confidence 0.5, fixed from the earlier local
  detector study before seeing ARC results. No search or tuning on results.
- For both appearance configurations retain raw online output, interpolation
  (upstream max-gap 10), and the exact upstream offline split/merge settings
  (cosine 0.85, max gap 300, no motion gate, split enabled).
- Compare person-only SORT, strict SORT, and the previously calibrated colour
  gate on the identical detections; replay stored SAM2 reprompt results using
  the same evaluator. SAM2 is not re-run. Colour calibration used Han-Ber4,
  so that baseline is not out of sample there.

## Scoring

Reuse the existing trusted reference masks/spans, player exclusions, IoU 0.5
greedy one-to-one observation matching and majority-per-track identity scorer.
Report correct and missed observations, wrong-identity frames, switches,
mixed tracks and fragments. Also report a fragmentation-sensitive globally
one-to-one correct-observation percentage; it is **not IDF1**, since incomplete
labels cannot supply a complete false-positive denominator. Do not compare
these percentages numerically with SportsMOT HOTA/IDF1.

References partly originate from propagation and were visually verified;
three short clips are a targeted screen, not independent full-match validation.
This scores local tracking IDs, not the full jersey/global-identity pipeline.
Post-processing is scored separately so offline linking cannot hide mistakes.

## Isolation and outputs

Use the existing pipeline Python without installing anything into its venv.
Extra dependencies (`lap`, `loguru`, and download helpers) are in this run's
private `deps/`; official vendor code is in `vendor/arcsort/`. The job sets
`CUDA_VISIBLE_DEVICES` empty, torch threads 2, interop 1, OpenCV threads 1,
BLAS threads 2, CPU affinity 0–1, nice 15, and idle I/O priority. CPU and shared
memory contention is still possible, but no GPU kernels/context are created.

Run root on Spark:
`/home/valentinweyer/projects/handball-computer-vision/runs/arc-sort-cpu-2026-10-21/`

It holds `protocol.json` (fixed before scoring), `manifest.json` (input hashes
and stride audit), per-frame CPU embeddings, raw/interpolated/final MOT rows,
matches, arbitration events, incremental `results.json`, `run.log`, and finally
`complete.json` including the final CUDA-uninitialized assertion. Input cache
hashes are checked again after processing. Existing artifacts are not deleted.

Reproduce from this worktree with the pinned vendor/weights already present:

```bash
ROOT=/home/valentinweyer/projects/handball-computer-vision/runs/arc-sort-cpu-2026-10-21
PY=/home/valentinweyer/projects/handball-computer-vision/.venv/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONPATH="$ROOT/deps:$ROOT/vendor/arcsort:$PWD/src:$PWD" \
taskset -c 0,1 nice -n 15 ionice -c 3 "$PY" -m experiments.arc_sort_cpu.run --root "$ROOT"
```

Feature reuse requires matching video, detection and weight hashes. Preserve the
locked protocol and the original results when exploring further configurations.
