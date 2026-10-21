# ARC-SORT CPU accuracy screen — 2026-10-21

## Scope

Targeted accuracy comparison on FelixClaar, Han-Ber4 and BHC-FAG, reusing
the existing trusted observation references and the production expanded
detector run on **every frame**. This is a short-clip screen, not full-match
or end-to-end global-player-ID validation. No speed measurements were made.

The experiment ran separately from the user's full inference job. CUDA was
disabled for the entire experiment. Extra Python packages were installed only
in a private artifact directory, not in the pipeline venv. CPU workers were
restricted to two cores with low scheduling and I/O priority.

Source: `experiment/arc-sort-cpu`, based on the existing oracle-study branch at
`161dcf1`. The new work only adds `experiments/arc_sort_cpu/` and this report;
no production tracker, team, identity or detector default was changed.

## Results

Completed all **33 cases**, CPU only; final assertion: CUDA uninitialized.

Each cell is **correct observations % / wrong-ID % / fragments per player**.
These percentages use different denominators; see the scoring notes below.

| Method | FelixClaar | Han-Ber4 | BHC-FAG |
|---|---:|---:|---:|
| SORT | 80.90 / 11.03 / 2.77 | 95.49 / 2.44 / 1.58 | 97.95 / 0.22 / 2.17 |
| Strict SORT | 83.65 / 7.63 / 3.08 | 95.28 / 2.48 / 1.67 | 97.89 / 0.22 / 2.17 |
| Colour-gated SORT | 83.35 / 8.16 / 2.69 | 96.97 / 0.22 / 2.00 | 98.03 / 0.14 / 2.17 |
| ARC motion only | 76.78 / 14.98 / 2.00 | 92.33 / 5.59 / 1.42 | 92.43 / 6.11 / 1.83 |
| ARC appearance, published preset | 87.89 / 2.72 / 1.83 | 93.06 / 5.00 / 1.25 | 95.80 / 2.82 / 1.67 |
| Published preset + interpolation | 88.68 / 2.74 / 1.83 | 93.50 / 5.19 / 1.25 | 95.90 / 2.84 / 1.67 |
| Published preset + interpolation + native offline | 86.39 / 5.25 / 1.83 | 93.50 / 5.19 / 1.17 | 95.41 / 3.34 / 1.67 |
| ARC appearance, local confidence 0.5 | 90.22 / 1.19 / 1.85 | 94.80 / 3.19 / 1.67 | 96.59 / 1.88 / 1.83 |
| Local confidence + interpolation | 91.26 / 1.13 / 1.92 | 95.23 / 3.17 / 1.58 | 96.73 / 1.90 / 1.92 |
| Local confidence + interpolation + native offline | 88.64 / 3.97 / 1.85 | 95.23 / 3.17 / 1.50 | 96.71 / 1.92 / 1.75 |
| Stored SAM2 reprompt | 95.01 / 0.78 / 1.38 | 99.91 / 0.00 / 1.00 | 98.88 / 0.12 / 1.42 |

Fragmentation-sensitive one-to-one correct-observation percentages:

| Method | FelixClaar | Han-Ber4 | BHC-FAG |
|---|---:|---:|---:|
| SORT | 63.09 | 88.95 | 90.91 |
| Strict SORT | 63.05 | 86.48 | 88.78 |
| Colour-gated SORT | 63.84 | 83.79 | 86.54 |
| ARC motion only | 69.75 | 88.56 | 89.26 |
| ARC appearance, published preset | 82.52 | 89.29 | 93.75 |
| Published preset + interpolation | 83.31 | 89.73 | 93.83 |
| Published preset + interpolation + native offline | 83.31 | 93.50 | 93.83 |
| ARC appearance, local confidence 0.5 | 80.32 | 84.53 | 91.23 |
| Local confidence + interpolation | 81.15 | 85.00 | 91.35 |
| Local confidence + interpolation + native offline | 85.48 | 87.91 | 92.90 |
| Stored SAM2 reprompt | 95.01 | 99.91 | 97.24 |

## Interpretation

- Appearance is essential here: motion-only ARC-SORT is materially worse on
  the first two clips. Its published sports OSNet is therefore included in
  the actual accuracy comparison, not silently omitted to make CPU use easy.
- On FelixClaar, the fixed local-confidence variant improves correct
  observations from strict SORT's 83.65% to 90.22% online and 91.26% with
  interpolation. Stored SAM2 is 95.01%. Online wrong-identity observations
  fall from strict SORT's 7.63% to 1.19%, but remain above SAM2's 0.78%.
- The native appearance-only offline split/merge is unsafe on this clip.
  It connects different white-kit players and lowers the local-confidence
  result from 91.26% after interpolation to 88.64% after merging. That is a
  distinct problem from whether ARC-SORT is useful for local tracking.
- On Han-Ber4, the local-confidence ARC online result is 94.80% correct,
  3.19% wrong among matched observations, against colour-gated SORT's
  96.97% / 0.22% and stored SAM2's 99.91% / 0.00%. After interpolation/merge,
  ARC is 95.23% correct / 3.17% wrong. The gap includes wrong links rather than only missing observations:
  recall for this arm is 98.35% after interpolation, while 72 matched
  observations have the wrong dominant identity. Restoring lower-confidence
  inputs could still change association, which this screen does not test.
- On BHC-FAG, local-confidence ARC is 96.59% correct / 1.88% wrong
  online, versus colour-gated SORT at 98.03% / 0.14% and stored SAM2 at
  98.88% / 0.12%. Interpolation raises ARC coverage slightly; native offline
  linking provides no net benefit.
- These results do not establish that box tracking cannot work. They show
  what the released ARC-SORT mechanisms do on these actual handball clips,
  rather than treating earlier SORT/ByteTrack results as a substitute.

## Visual review

Two contact sheets compare online ARC with local confidence, its native
offline result, and stored SAM2 at the same frames. The coloured reference
outlines are red/cyan; green boxes are predictions. `T` is the native tracker
ID. `R` is its dominant reference identity over the clip, a diagnostic mapping
used by the scorer, **not a jersey number or a global player prediction**.

- `felix_contact.jpg`: two players remain in close contact. Around frame 95,
  ARC's boxes exchange/overlap identities; later views recover the mapping.
- `felix_offline_merge.jpg`: the native offline merger assigns one former
  white player's identity to another white player, although the online
  version has separate IDs. Frames 185 and 205 make the difference clear.

The scorer also finds some very short mismatches, and reference quality is
not perfect. No new reference labels or trusted spans were edited.

## Reproducibility and qualifications

All protocol settings were saved before scoring in `protocol.json`. The
manifest records hashes of the videos, input detection caches, weight file,
and reference masks. Every detection cache has a populated offset slot for
every decoded frame: 249 FelixClaar, 199 Han-Ber4, 500 BHC-FAG (948 total).
Only goalkeeper/player classes 1/2 enter candidates and box baselines.

ARC-SORT vendor revision:
`2a580c592c838bce99b89a6a2dbd7ad6f50c84be`.
The official Deep-EIoU sports OSNet checkpoint SHA-256:
`8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd`.

The original SportsMOT preset is one configuration. A second, declared
before scoring, uses activation/new-track confidence 0.5, inherited from the
previous local detector study. Neither is selected separately per clip or
tuned after seeing these results. Motion-only uses the preset. Box baselines
use their earlier fixed settings, including the colour gate 0.28 calibrated
on Han-Ber4; its Han score is in sample.

Appearance preprocessing and network weights match the released sports
model, but inference is FP32 CPU instead of the upstream FP16 CUDA path.
Features use upstream's per-frame embedding-cache interface. Cache misses
raise an error so there is no silent GPU fallback or missing-feature run.
Association, camera-motion compensation, arbitration, interpolation and
split/merge use the pinned vendor implementation. Offline settings follow
its reproduction script: interpolate at gap 10, then merge at cosine 0.85,
gap 300, no motion gate, split enabled.

**Detection confidence limitation:** the existing stride-1 production caches
discard scores below 0.3. ARC-SORT normally uses low-confidence boxes down
to 0.1, so this does not test its complete low-confidence recovery range.
Lower-floor, stride-1 detector runs are a separate follow-up; no new GPU
detection was launched alongside the user's inference job. The CPU-only
screen must not be presented as a ceiling on ARC-SORT accuracy.

The shared evaluator matches boxes to trusted reference-mask boxes at IoU
0.5. Correct-observation percentages include missed observations in the
denominator; wrong-ID percentages use matched player observations. Each
track is assigned its dominant reference identity for the historical score,
so fragmentation is reported separately. An additional globally one-to-one
correct-observation metric penalizes fragmentation; it is not IDF1, because
the labels do not supply complete false-positive coverage.

The references are short and partly generated by mask propagation, then
visually checked. Han-Ber4 in particular is partly circular with SAM2.
These are local IDs; no jersey reader, player registry or smart-global-ID
linker was run. Removing downstream masks and integrating a different local
tracker still require their own assessment before any production switch.

## Artifacts on Spark

Root:
`/home/valentinweyer/projects/handball-computer-vision/runs/arc-sort-cpu-2026-10-21/`

It contains the locked protocol and input manifest, downloaded official
weights/vendor source, private dependencies, CPU feature caches, raw MOT
outputs, interpolation and final outputs, arbitration events, per-frame
matches, results JSON, contact sheets, log and completion record.

Reproduction commands and the input contract are in
`experiments/arc_sort_cpu/README.md` on the experiment branch.

## Decision

Keep SAM2 as the production reference. ARC-SORT is a measured improvement
over the earlier box trackers on the hardest clip, but this released
configuration does not preserve SAM2 accuracy across the three clips and
is worse than the existing colour-gated baseline on two. Do not enable its
native offline merger based on the published aggregate result. A further
trial should first restore the 0.1–0.3 detection range at stride 1, keep
online tracking separate from global-player linking, and use longer fresh
contact-heavy footage before considering a production change. The low-score
input limitation and short/partly propagated references prevent treating
this screen as a final rejection of ARC-SORT.
