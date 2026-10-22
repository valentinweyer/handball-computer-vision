# ARC as an opt-in pipeline backend

`process_video --tracker arc` runs the complete cached pipeline: every-frame
RF-DETR, ARC local tracks, the existing IdentityManager/PlayerRegistry with
PRTReID, per-video team evidence, PARSeq reads, number voting/linking, court
coordinates, result bundle, and the normal result renderer. SAM2 remains the
default. The old custom overlap repair is not used.

## Assets and command

Use the existing environment. ARC additionally needs lap, loguru and PyYAML
(the optional `arc` dependency group), an external ARC source checkout pinned
to `2a580c592c838bce99b89a6a2dbd7ad6f50c84be`, and the evaluated official sports
OSNet checkpoint. The adapter checks its SHA-256
`8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd`.
There are no automatic downloads and no environment synchronization.

```sh
python -m scripts.process_video data/raw/match.mp4 --tracker arc \
  --arc-root /path/to/pinned/arcsort \
  --arc-weights /path/to/sports_model.pth.tar-60
python -m scripts.render_result runs/bundles/match_arc
```

Default work/bundle names gain `_arc`, preserving existing SAM2 results. Explicit
output paths containing another tracker's bundle are refused. Resuming a work
directory with another tracker/config requires a fresh directory or an explicit
forced tracking pass. A stride-five cache cannot serve ARC: select a fresh work
directory or explicitly regenerate scanning. Genuine empty frames are legal;
verification uses stride, scored-frame count, total count and offsets.

`--arc-device cpu` (default) selects CPU OSNet; `cuda` is available explicitly.
Native ARC association is CPU-based. `--device` still controls team/PRTReID/OCR
models and scanning. CPU OSNet is not a claim that the complete pipeline needs
no GPU or is faster. The source adapter uses local inference and blocks ARC's
implicit CUDA fallback.

## Fixed configuration and identity

Use the evaluated SportsMOT config with high/new thresholds `.5/.5`. Native
ambiguity/arbitration remains enabled. No custom repair, interpolation, offline
merge, team gating, or threshold tuning is applied. The existing detector cache
floor `.3` truncates ARC's nominal `.1–.3` BYTE low-confidence range; this is the
same limitation as the earlier comparison.

ARC's OSNet features describe local associations. The existing PRTReID and
jersey number logic remain downstream identity evidence. ARC has no access to
identity-team labels or jersey votes. Raw team observations stay frame-local.

The bundle's existing `track_id` retains its pre-number-link registry meaning;
`player_id` is the downstream linked identity. ARC's *native* source local ID is
additionally preserved in `local_tracks.npz`, with arrays `frame`,
`local_track_id`, `identity_id` (before number aliases), one row per output.
The run JSON points at it; the bundle copies it as an optional file. Existing
table schemas and all SAM2 bundles remain readable. Join by frame/identity_id
and follow existing number-link provenance when diagnosing coverage. Do not
mistake a native track's rebirth for a new person or a number link for tracker
continuity.

Court-not-visible frames pause tracking and retire its active mappings. Resuming
resets motion/appearance history while advancing the local-ID namespace, so a
native reset cannot reuse an earlier local ID. Existing global appearance ReID
may still reconnect identities across that gap.

## Numbers and missing masks

ARC emits boxes, not player segmentation. No rectangular silhouette is stored
or claimed as a mask. Empty masks in the geometry bundle mean absent masks.
Court feet use the existing box bottom-centre convention.

Number ownership is deliberately conservative: a number crop must belong to
exactly one raw person detection (more than half its area inside the person),
then that owner must map unambiguously to an ARC output box (IoU >= .5,
runner-up margin >= .1). Competing unconfirmed detections can veto ownership.
The existing one-to-one selection, crop readability, reader, vote and link
safeguards still apply. This box policy can read fewer numbers during overlap
than SAM2's mask policy. That is an explicit baseline limitation to measure,
not justification to relax its gates in this integration.

Direct cached renders use `render_full_pipeline --tracker arc`; number boxes
come from `--number-detections` or the every-frame detection cache, never a
hosted detector. `--numbers off`, `--no-video`, cached redraw, and result bundles
are supported. ARC redraw uses native video pixels, matching its tracking pass.

## Validation and stop condition

Tests cover strict dense-cache loading, real empty frames, source/identity ID
separation, reset ID uniqueness, ambiguous number ownership including an
unconfirmed competitor, absent-mask geometry, default SAM2 selection and resume
protection. Native parity is checked against the earlier frozen ARC output.
A ten-second CPU-only end-to-end smoke uses cached real detections, team model,
keypoints and motion with real PRTReID and PARSeq, then builds/loads/renders a
bundle. It is integration proof, not an accuracy measurement.

Stop after this baseline is reviewable. Team-aware association and full-match
quality/coverage validation are separate follow-up work; production stays SAM2.


## Spark validation environment

No package was installed or upgraded. Run from this feature worktree with its
src directory and root on PYTHONPATH; also include the existing isolated
dependencies at runs/arc-sort-cpu-2026-10-21/deps under the primary repository.
Use the primary repository's .venv/bin/python directly. This worktree's ignored
default source/weight paths link to the already evaluated assets. On another
checkout, pass --arc-root and --arc-weights or prepare those asset paths.

The saved integration artifacts live in runs/arc-pipeline-integration-2026-10-22/
under the primary repository. The ten-second test produced 2,413 track
observations, 65 accepted OCR reads and five resolved jersey-number votes.
All 250 rendered frames decoded successfully. The source-local-ID mapping
matched every bundle track row; duplicate identities/numbers per team were
absent. The geometry contains no fabricated mask pixels. The prior frozen
full-run detector and team-model hashes remain unchanged. Native parity also
verified that a reset starts a fresh local-ID namespace.
