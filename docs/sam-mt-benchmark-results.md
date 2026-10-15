# SAM-MT native comparison — 2026-10-15

**Decision: stop this comparison and keep production SAM2, at the user's
request.** SAM-MT lost substantially on the two completed clips. The suite did
not complete its third clip or repeated runs. SAM-MT remains installed as an
experiment; no production tracker or default was changed.

## Completed results

Correct means the share of trusted reference player detections found with the
right identity, using the existing box-IoU scorer. These are **native seed-once
models with identical point prompts**, not the periodic-reprompt production
pipeline.

| Clip | SAM-MT correct | SAM2.1-L correct | SAM-MT FPS | SAM2.1-L FPS |
|---|---:|---:|---:|---:|
| FelixClaar, 249 frames / 13 seed targets | 69.87% | 93.72% | 3.83 | 1.76 |
| Han-Ber4, 199 frames / 14 seed targets | 65.19% | 94.15% | 1.22 | 1.72 |

On Felix, SAM-MT was 23.85 percentage points worse with 2.18× measured
throughput. On Han-Ber4, it was 28.96 points worse and slower (0.71× throughput).
These are single warmed observations on a shared machine under memory pressure;
they do not establish a repeatable speed ratio.

| Clip / model | Matched reference detections | Wrong-ID frames | Mixed tracklets | Switches |
|---|---:|---:|---:|---:|
| Felix / SAM-MT | 1742 | 63 | 5 | 19 |
| Felix / SAM2 | 2310 | 58 | 6 | 9 |
| Han / SAM-MT | 1720 | 216 | 8 | 14 |
| Han / SAM2 | 2172 | 0 | 0 | 0 |

Full counts and per-reference timelines are in each run's `score.json`.
Box-match misses and switches are not automatically verified body losses or
swaps. The planned dense mask review of Felix's teammate overlap was not run.

## Protocol and limitations

- Official SAM-MT at source `55952679e062feb05004b598bb835d4a7a8205a1`, with the
  pinned official checkpoint; SAM2.1 Hiera-L from the existing upstream checkout.
- Both ran on NVIDIA GB10 with BF16 autocast, 1024 input, compilation disabled,
  strict checkpoint loading, and GPU model/frame/state storage.
- Identical cached JPEGs and first-frame RF-DETR boxes. Each target received two
  clicks on its box centreline at 30% and 50% of box height. These are automatic
  detector-derived clicks, not manually verified foreground points.
- SAM-MT used a shared target group; output channels were mapped back to the
  corresponding seed IDs. SAM2 received the same points separately per ID.
- Each arm ran in a fresh process, warmed on a separate 21-frame state, then
  initialized a fresh measured state. Timing covers frames after frame zero,
  native propagation, original-resolution masks, shared edge-fragment filtering,
  CPU mask transfer and boxes. Initialization and serialization are separate.
- No detector checkpoints, new entrants, retirement, re-ID, OCR or rendering.
  Consequently these controls differ from the [production tracking
  benchmark](tracking-evaluation.md), where SAM2 uses periodic box reprompting.
- The unchanged scorer and trusted spans were reused. Han's denominator excludes
  bench/referee reference IDs 12 and 14; its SAM2-derived reference is not an
  architecturally independent quality check. Felix uses the independent SAM+Cutie
  reference with its existing truncated trusted spans.

## Incomplete work and memory pressure

The BHC-FAG SAM-MT process died with SIGKILL after warmup, before writing a
measured replay or score. The suite then exited; none of its second or third
repeats ran. No BHC-FAG result or three-repeat median should be inferred.

Inspection found about 112 GiB of 121 GiB RAM used, nearly all 15 GiB swap used,
and NVIDIA driver out-of-memory messages. This demonstrates severe memory
pressure, but does not by itself prove who sent SIGKILL. Other GPU services were
left running. The timing decline between clips must not be attributed solely
to model architecture.

## Evidence and reproduction

Artifacts: `runs/sam_mt_benchmark_20261015/paired/`, containing `summary.json`,
per-arm `manifest.json`, `replay.npz`, `score.json`, `matches.json`, `frames.json`,
and logs. The failure is recorded in `runs/sam_mt_benchmark_20261015/suite.log`.
The four completed runs end in `_r1`; validation runs are separate and excluded.

The retained [benchmark runner](../experiments/sam_mt/benchmark.py) and
[suite runner](../experiments/sam_mt/run_suite.py) reproduce this protocol. Do
not rerun automatically: the user explicitly stopped the comparison. Any future
run must use a new output directory and account for available memory.
