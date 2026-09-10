# EfficientTAM lifecycle investigation — 2026-09-09

**Follow-up 2026-09-10:** the minimal factory integration and trained three-clip
comparison are now implemented. See [paired results](efficienttam-paired-results.md)
for the executed evidence. The initial investigation below records what was
known before that follow-up.

**EfficientTAM-S at 1024 passes the source/API lifecycle gate. It is not yet a
verified quality or speed replacement.** The smallest integration is a predictor
construction seam in the existing driver, preserving the manager and output path.

Started from clean `feat/team-classification` at `9952d0c`. Production code,
defaults, weights, dependencies, detections and labels were not changed. The only
executable addition is the isolated [lifecycle audit](../experiments/efficienttam/audit_lifecycle.py).

## Evidence and limits

Pinned source:

- [EfficientTAM abcd061](https://github.com/yformer/EfficientTAM/tree/abcd061ebd3cc6e7527d152d75b890126aaa53f6).
- Installed SAM2: `2b90b9f5ceec907a1c18123530e92e794ad901a4`, clean upstream tree.
- Project: `9952d0c`; GB10, PyTorch `2.9.1+cu130`, CUDA available.

Ten lifecycle methods have identical Python ASTs after normalizing only
`EfficientTAM` to `SAM2` in string literals: initialization, ID allocation, box
and mask prompting, propagation preflight and iteration, removal, reset,
per-frame prompt clearing, and original-resolution output. This comparison
**does not cover the neural computation** or establish model equivalence.

An executed GPU smoke test used the real **untrained** Small model, 1024 input,
BF16 prompts/propagation, compilation disabled, eight existing Han-Ber4 JPEGs,
and three synthetic boxes with nonconsecutive IDs. It passed:

- Late addition preserving existing object memory.
- In-place correction retaining history and other objects.
- Middle-slot removal remapping indices without losing survivor memory.
- Same-ID fresh reset, with the survivor unchanged.
- Single-object in-place correction fallback and resumption.
- Last-object removal and subsequent reseeding.

Survivor containers, frame entries and stored memory tensors were checked for
preservation. Repeated chunk boundaries, returned ID order, finite
original-resolution logits, the existing mask filter and mask-derived box shapes
were checked too. Local result: `runs/efficienttam_lifecycle/audit.json`.

The probe exercises predictor operations directly, not the complete
`drive_sam2`/`TrackManager`/jersey pipeline. Random-weight masks say nothing about
player identity or usable segmentation. Progress-bar rates are not benchmarks.
No trained checkpoint was downloaded or evaluated. Survival through handball
occlusion and equivalent tracking quality remain **unverified**.

Reproduce in a fresh process, with a checkout at the pinned revision:

```bash
PYTHONPATH=/path/to/EfficientTAM uv run --no-sync python \
  -m experiments.efficienttam.audit_lifecycle \
  --frames data/cache/frames/Han-Ber4_cached
```

## Lifecycle mapping

The pinned [predictor source](https://github.com/yformer/EfficientTAM/blob/abcd061ebd3cc6e7527d152d75b890126aaa53f6/efficient_track_anything/efficienttam_video_predictor.py)
supports the current driver's operations:

| Requirement | Mechanism and consequence |
|---|---|
| Video state | `init_state` accepts the same JPEG-directory and offload arguments (49–106). Frame features are shared; object inputs, outputs and histories have separate stores. |
| Later entrant | `_obj_id_to_idx` always allows new IDs (127–158). A box at the checkpoint establishes that object's conditioning frame without a global reset. |
| Box correction | `add_new_points_or_box` accepts pixel XYXY corners, scaled into prompt labels 2/3 (170–302). `clear_old_points=True` clears frame-local prompts, not history. Previous logits remain a correction prior. |
| Consolidation | Preflight encodes temporary prompts into object memory before propagation (489–552). Preserve the current inclusive checkpoint restart and duplicate-frame skip initially. |
| Propagation | Independent object loop with `batch_size=1`; returns `(frame_idx, obj_ids, logits)` (555–640). Object-dependent cost remains. |
| Removal | `remove_object` discards target stores and reindexes survivors (876–963). Public IDs survive; array slots can change. |
| Body-swap reset | Remove target, then box-prompt the same ID at the checkpoint. Its history is fresh; other players retain history. The target moves to the last output slot. |
| Last object | Removal clears all object state. Preserve the driver's in-place correction fallback when a reset targets the only remaining object. |
| Outputs | `(N,1,H,W)` original-resolution logits. Keep threshold `>0`, `masks_from_logits`, `sv.mask_to_xyxy`, returned ID ordering and lazy `read_frame` in `Sam2FrameResult`. |

Two boundaries remain explicit. The predictor rejects propagation with zero
prompted objects; the existing driver has no empty-session wait-and-reseed loop.
The probe proves reseeding at the predictor level, not application recovery from
an entirely empty court. Also, optional `clear_non_cond_mem_around_input=True`
calls an undefined `_clear_obj_non_cond_mem_around_input` in this EfficientTAM
revision. It defaults to false; keep it false rather than enabling an untested
memory repair.

## Exact candidate and construction traps

First arm: **`configs/efficienttam/efficienttam_s.yaml` + `efficienttam_s.pt`**,
1024, seven mask memories, eager inference. `_s_1` and `_s_2` select different
efficient cross-attention implementations and require their respective
checkpoints. They are later, separately named arms, not aliases for plain Small
or an interchangeable interpretation of the paper's S/2 name.

The [Small config](https://github.com/yformer/EfficientTAM/blob/abcd061ebd3cc6e7527d152d75b890126aaa53f6/efficient_track_anything/configs/efficienttam/efficienttam_s.yaml)
sets **`compile_image_encoder: true`**. `vos_optimized=False` alone is insufficient:

```python
build_efficienttam_video_predictor(
    "configs/efficienttam/efficienttam_s.yaml",
    ckpt_path="/path/to/efficienttam_s.pt",
    vos_optimized=False,
    hydra_overrides_extra=["++model.compile_image_encoder=false"],
)
```

Before a trained comparison:

- Use a pinned external checkout without dependency resolution. Upstream
  `setup.py` pins `supervision==0.25.0` and old hub/UI packages conflicting with
  this environment. The smoke succeeded through `PYTHONPATH` without installing
  anything. Preserve `uv run --no-sync`.
- Run each backend in a fresh process. Both packages initialize Hydra's global
  config root on import only if uninitialized; importing SAM2 first can leave
  EfficientTAM configs undiscoverable. Keep builder imports lazy. Avoid clearing
  global Hydra inside reusable pipeline code.
- The smoke warned that `efficient_track_anything._C` is missing and skipped
  low-resolution hole filling although the builder requests area 8. `sam2._C`
  is also absent in the installed baseline. Record requested **and effective**
  post-processing and capture warnings in both trained runs. Do not silently
  install an extension for one arm. The separate project OpenCV edge-fragment
  filter did execute in the smoke.
- Require a real checkpoint and save its SHA-256. The builder accepts
  `ckpt_path=None`; a benchmark runner must reject it so this smoke cannot be
  mistaken for a model result.

## Smallest integration to implement next

1. Add one optional predictor factory argument to
   `src/handball_cv/tracking/sam2_driver.py`. With it absent, lazily construct SAM2
   exactly as today. With it supplied, construct EfficientTAM from explicit
   config/weights. Leave the seed, propagation and checkpoint action sequence
   unchanged. Preserve RF-DETR, `TrackManager`, `PlayerRegistry`, per-object
   memory, additions/removals, corrections/resets and `Sam2FrameResult`.
2. Extend the tracking experiment runner with backend/config/checkpoint selection,
   separate outputs and a run manifest. Create the output directory **before**
   event JSON writing (the current runner does this afterwards). Add startup and
   complete-loop timing, checkpoint timing and active-object counts. No second
   lifecycle implementation or general tracking framework is needed.
3. Add explicit named replay paths and optional per-frame match export to
   `scripts/evaluate_tracker_identity.py`. Reuse `Sam2ReplayTracker`,
   `load_reference`, `run_tracker` and `score`, preserving historical matching.
   Add continuity analysis alongside existing metrics, with fixtures for clean
   fragmentation, same-team exchanges and missing-frame gaps under one reused ID.

Keep SAM2 the default. Verify the default factory path against fresh SAM2 replay
and events, then run a trained lifecycle case through the driver before the
paired clips. Full-pipeline consumers can use the same factory after this gate.

One workload mismatch matters: `run_sam2_reprompt_tracker.py` passes unfiltered
cached classes and no separate re-ID encoder; `render_full_pipeline.py` uses
`person_detections` and PRTReID. The three historical caches have two classes, so
this did not corrupt their comparison. Filter Melsungen's multiclass cache to
people. Declare the identity configuration and hold it fixed within each pair:
historical tracking configuration first, deployed PRTReID/PARSeq application as
a separate pair, without weakening its defaults.

## Paired evaluation: continuity first

Regenerate SAM2 outputs at the same project revision as the candidate. Recorded
correct-detection baselines are **95.0 / 89.2 / 98.9%** on FelixClaar, Han-Ber4 and
BHC-FAG. Old `runs/sam2_reprompt/<stem>.npz` files predate the lifecycle refresh;
`*_policy_reprompt.npz` are newer historical artifacts. Never overwrite either.
Han-Ber4's **2.17 FPS**, with **87.9% model inference**, describes the recorded
tracking loop, not the deployed OCR/re-ID/rendering application.

Freeze detections, fitted team model, checkpoint interval 10, frame order and
spacing, mask filtering, original output resolution, precision, re-ID settings,
exclusions and trusted spans. Give each backend a fresh state and manager.
Natural differences in manager actions are outcomes to record; do not suppress
them by forcing both models to receive the same manager decisions.

| Video under `data/raw/` | Reference under `source/` | Labels under `runs/tracklet_labels/` |
|---|---|---|
| `FelixClaar.mp4` | `.FelixClaar_ref_masks` | `FelixClaar_reference/answers.txt` |
| `Han-Ber4_cached.mp4` | `.Han-Ber4_sam2_masks` | `HanBer4_sam2_reference/answers.txt`; exclude 12, 14 from player scoring |
| `BHC-FAG_window_cached.mp4` | `.BHC-FAG_window_ref_masks` | `BHC-FAG_window_reference/answers.txt` |

Reuse Felix's detection/model caches in `outputs/team_comparison`; Han-Ber4's
detections in `outputs/team_dataset` and model in `outputs/team_correction_mcbyte`;
BHC window detections in `outputs/team_dataset` and the full-match fitted model
in `outputs/team_confidence_v2`. Check offsets against first/last reference frames.

The current scorer assigns each predicted tracklet its dominant reference
identity. Its switch count follows reference-identity changes **within a predicted
tracklet**, not predicted-ID changes for one reference player. Clean fragments
can all score correct. Distinct-ID fragmentation also misses repeated
loss/recovery under one reused ID. Add:

- **Per-player timelines:** each trusted, scoreable frame gets a matched raw ID
  or missing, plus a generation counter for retirement/revival or memory reset.
  Keep downstream corrected identity separate so number aliases cannot hide
  tracking failures.
- **Continuity:** for each trusted visible span, report longest uninterrupted
  correct run, missing-run counts/durations, predicted-ID transitions and recovery
  delay. Fix a one-to-one identity mapping from a verified pre-event frame for
  each overlap; retain it through the event. Per-frame remapping or assigning
  each new fragment its own identity hides swaps. Split scoring at untrusted
  reference boundaries.
- **Same-team overlap events:** inspect both players frame by frame before,
  during and after contact, using verified team/person labels rather than
  tracker team-switch diagnostics. Record wrong-person duration, recovery,
  fragmentation, mask usability and correction burden. Include some events
  where both models agree, because correlated failures are possible.
- **Complete occlusion and entrants:** existing references do not cover every
  later entrant or provide visible masks during complete occlusion. Add a small
  event ledger of visible/occluded/out-of-frame/untrusted intervals and pre/post
  identity. Do not score absence of a mask during complete occlusion as a visible
  miss or invent interpolated truth. Check correct identity on reappearance and
  unintended handoffs. Manually review the BHC pileup around frames 281–345;
  several trusted identity spans end at 280 there.

Retain recall, wrong identity, correct-detection percentage, mixed tracklets,
switches and fragments per clip with counts and denominators. Han-Ber4's
SAM2-built reference is corroborating; Felix/BHC use the more independent
SAM+Cutie references, also limited by sparse verification. Current references
cannot establish statistical equivalence of complete player trajectories.

The old provisional “within one percentage point” rule is insufficient for this
request. New same-team handoffs, longer wrong-person episodes or increased
visible-span fragmentation require explanation and block replacement even when
aggregate correctness improves. Report corrections/resets per player-minute;
downstream repair must not conceal frequent visual tracking failures. Keep 2×
loop throughput as a useful speed target, not permission to sacrifice identity.

## Throughput protocol and stopping point

After trained lifecycle checks pass, run three paired repeats per short clip,
alternating backend order with no competing GPU job. Warm up on separate state,
then initialize a fresh state/manager. Report cold build, frame preload and seed
cost separately. Physically bound JPEG caches; `max_frames` does not bound preload.

Synchronize CUDA around the complete measured interval. Include propagation, CPU
masks, filtering, boxes, manager updates, checkpoint image reads, detection cache
lookup, decisions and prompts. Count completed unique frames, including empty
outputs. Exclude the separate frame-0 seed and repeated chunk boundaries from the
numerator while retaining boundary work in timing. Exhaust the generator:
checkpoint work happens after a yielded frame, so individual `next()` latency
can misattribute it. Profile synchronized stages in a separate diagnostic run.

Report FPS, total wall time, median/p95 frame latency, checkpoint latency, active
object distribution, CUDA allocated/reserved peaks and process/system memory.
Keep all selected players active, including goalkeepers; fewer targets is not a
speed improvement. Report natural full-loop occupancy and a separate controlled
1/4/8/12/16-object propagation sweep. The manager caps later additions at 20;
do not silently impose a smaller backend cap. Use the 60-second Melsungen window
for resource growth/lifecycle load, not independently scored accuracy.

Then time full RF-DETR/team/PRTReID/PARSeq/render/write execution separately.
Cached detector-loop FPS and replay-scoring FPS are not application throughput.

**Next executable milestone:** the factory/runner/replay changes, an actual
`efficienttam_s.pt` load, a trained lifecycle case, and the three-clip pair.
Stop if lifecycle fails or continuity regresses; retain SAM2. Same-weight
compilation and EdgeTAM reset/reseed remain rejected on recorded evidence.
DAM remains a separate later robustness experiment with no verified integration.
