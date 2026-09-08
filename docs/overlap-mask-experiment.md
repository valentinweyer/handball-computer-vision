# Overlap-mask experiment: guarded mask construction and results

Full record of the guarded overlap-mask fallback experiment, moved out of
`CLAUDE.md` on 2026-08-28 to keep the always-loaded project instructions small.
`CLAUDE.md` retains the architectural rule, the usability gates, the headline
finding and its anti-claim caveats; everything below is the construction detail,
the numeric gate table, and the per-clip measurements behind them.

Diagnostic source: `notebooks/render_mask_team_comparison.py`.
Implementation: `notebooks/mask_team_features.py`.

## Guarded mask construction (detail)

Implemented in `notebooks/mask_team_features.py`.

MCByte's `_last_mask_output` is spatially aligned to the current frame, although it is produced before current-frame association using prior track state. It contains:

- `masks`: boolean `(K, H, W)` masks.
- `tracklet_mask_dict`: stable tracker ID to mask-row mapping.
- `mask_avg_prob_dict`: tracker ID to average winning-mask probability.

Do not select a target mask using tracker ID alone. A tracker association error would then become a confident team-color error.

The implementation:

1. Assigns current detector boxes to current masks one-to-one using Hungarian matching on mask/box geometry.
2. Separately checks that the spatial assignment agrees with MCByte's tracker-ID-to-mask mapping.
3. Abstains on disagreement or ambiguous spatial matching.
4. Intersects the target mask with the torso crop.
5. Erodes the target mask boundary.
6. Dilates nearby masks and removes that uncertain neighbor boundary.

The reliable pixels are conceptually:

```text
safe jersey pixels = torso ROI
                   AND eroded target mask
                   AND NOT dilated neighboring masks
```

Cutie masks are mutually exclusive already; neighbor dilation creates the useful uncertainty margin.

Current mask gates:

- Mask average confidence: at least 0.60.
- Fraction of target mask inside current box: at least 0.80.
- Mask fill of current box: at least 0.08.
- Spatial assignment score: at least 0.25.
- Assignment margin over alternatives: at least 0.02.
- Safe torso coverage: at least 0.15.
- Retained fraction of target torso mask: at least 0.50.
- Safe pixels before resize: at least 80.
- Safe pixels after 32x32 resize: at least 24.
- Erosion radius: 2.5% of the smaller torso dimension, clamped to 1-5 pixels.
- Neighbor dilation radius: erosion radius + 1, capped at 5 pixels.

The diagnostic also used a stricter MCByte mask-creation overlap threshold of 0.20 instead of the package default 0.60, so a heavily overlapped box is less likely to initialize a contaminated SAM mask.


## Measured results (detail)

Diagnostic: `notebooks/render_mask_team_comparison.py`.

It performs frame-local raw box-color classification and frame-local guarded mask-color classification on identical detections. Tracking supplies masks and diagnostic IDs only. There is no temporal team vote in this experiment.

Overlap begins at torso contamination 0.05. A mask is considered usable by the real team-state policy only when:

- Geometry is accepted.
- Team confidence is at least 0.30.
- Effective mask observation quality is at least 0.40.

### FelixClaar

- 249 frames.
- 2,598 field-player detections.
- 562 overlap detections.
- 259 geometrically accepted masks: 46.1% acceptance.
- 159 observations were unusable through the box-overlap gate but usable through the mask path.
- Four geometric mask classifications changed team.
- Two changes passed the real confidence/quality gate and look visually plausible in extreme overlap frames.
- There is no substantial manually labeled Felix overlap set, so do not claim a numerical accuracy improvement from Felix.

Primary rejection counts:

- Tracker/mask disagreement: 133.
- Unassigned: 71.
- Low box coverage: 57.
- Ambiguous spatial assignment: 31.

### Han-Ber4

- 199 frames.
- 2,206 field-player detections.
- 464 overlap detections.
- 245 geometrically accepted masks: 52.8% acceptance.
- 112 observations were unusable through the box-overlap gate but usable through the mask path.
- Six geometric mask classifications changed team.

Manual overlap ground truth is very small:

- Five substantial manually labeled overlap crops, all from team B.
- Three had geometrically accepted masks.
- Raw, mask, and chosen classification were all correct on those five.

The independent Han-Ber reference-mask evaluation is more useful:

- 339 overlap detections matched to existing SAM2 reference identities at IoU >= 0.65.
- Raw box-color accuracy: 95.28%.
- If every geometrically accepted mask is used, accuracy appears to rise to 96.46%: five corrections and one regression.
- Those six changes all had low team confidence.
- Under the actual confidence/quality gate, 138 mask observations were selected, none changed the team label, and accuracy remained 95.28%.

Interpretation: the demonstrated value is recovering additional qualified same-team evidence, not yet improving qualified per-frame accuracy on Han-Ber. Do not quote the 96.46% geometric result without the confidence-gating caveat.

### Runtime cost

On the available NVIDIA GB10 machine, the complete diagnostics ran at roughly 3.8-4.8 frames per second. Each 199/249-frame clip took about 52 seconds. SAM/Cutie mask propagation dominates this experiment; the color classifier itself is cheap.

