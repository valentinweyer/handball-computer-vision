# ARC team-confirmed overlap repair — 2026-10-21

Implemented on `experiment/arc-team-repair`, based on the CPU ARC-SORT
experiment at `def6081`. The experiment flag is `--team-overlap-repair`, default
off. Production tracking, team classification and global identity assignment
are unchanged. This branch has not been merged.

## Result

One accepted lost-partner repair on Han recovered **41 correct player-frame
observations, with zero previously correct observations lost**. Felix and BHC
were byte-identical with the option enabled. All three clips reproduce the
original ARC output byte-for-byte when disabled.

| Clip / processing | Correct before | Correct after | Wrong identity frames before → after | Gained / lost correct |
| --- | ---: | ---: | ---: | ---: |
| Felix / raw | 90.22% | 90.22% | 26 → 26 | 0 / 0 |
| Han / raw | 94.80% | 96.58% | 72 → 31 | 41 / 0 |
| BHC / raw | 96.59% | 96.59% | 94 → 94 | 0 / 0 |
| Felix / interpolated | 91.26% | 91.26% | 25 → 25 | 0 / 0 |
| Han / interpolated | 95.23% | 97.01% | 72 → 31 | 41 / 0 |
| BHC / interpolated | 96.73% | 96.73% | 95 → 95 | 0 / 0 |

“Correct” uses the existing dominant-identity-per-track scorer and divides by
trusted reference player observations, including misses in that denominator.
It is not HOTA or IDF1. The paired check compares which individual
`(frame, reference player)` observations are correct before and after.
Han's separate globally one-to-one correct score also improves, from 84.53%
to 86.30% raw and from 85.00% to 86.78% interpolated. Fragmentation, detector
coverage and box coordinates remain unchanged. Native interpolation fills
gaps of at most ten frames; ARC's appearance-only offline merger is excluded.

This is one development event, not a held-out result or a general guarantee of
no regression. It does not establish parity with SAM2, whose stored Han score
is 99.91% on this reference set.

## What the repair does

Team observations are computed independently on raw, every-frame detections.
The existing per-video colour model is reused on CPU. Qualified observations
require a field-player detection, the existing minimum confidence of 0.30,
accepted crop quality of at least 0.30, torso contamination at most 0.05,
other-detection IoU below 0.10, and an unambiguous geometric match to an ARC box
(IoU at least 0.50 with a margin of at least 0.10). Repaired identities never
feed back into the classifier or observation cache.

The pure repair layer supports:

- **Two live tracks exchanging teams:** opposite trusted pre-contact teams,
  mutually reversed post-contact teams, no third overlapping track, and a
  unique geometric crossing boundary. It rewrites IDs from that boundary.
  This path is exercised by synthetic regression tests; none of these three
  real clips qualifies for it.
- **A lost partner's continuation being taken over:** one track ends in contact,
  the survivor changes to that track's trusted team, and its next three boxes
  provide a spatial bridge of IoU at least 0.30 to the lost track's final box.
  A competing plausible lost partner or continuation vetoes the repair,
  including one with missing team evidence. The repair starts at the first
  clear observation supporting the subsequently confirmed team, leaving the
  ambiguous interval alone.

Anchors require at least five qualified observations, at most ten votes and
90% agreement, within forty frames before or forty-five frames after contact.
At least 95% of all later qualified team reads must remain consistent. One
accepted repair per involved raw track avoids untested compositions. Returning
lost tracks, same-team contacts, missing evidence, uncertain geometry and later
contradictions abstain. Only ID fields change; there are no new boxes, dropped
rows, duplicate IDs within a frame, or changes to raw team observations.

## The Han case and the boundary tradeoff

Raw track T2 initially follows a red-shirted player, then follows the white
player previously tracked as T11. T11 ends at frame 128 during contact.
The first clear qualified white-team observation on T2 is frame 142; ten
consistent votes confirm it through frame 151. The accepted repair labels
frames 142–182 as T11, backfilling from that confirmation to the first clear
supporting observation. Frame numbers here are zero-based.

The approach cannot safely choose every route through an overlap. An earlier
development revision relabeled immediately after T11 disappeared and gained
52 correct observations but lost two at frames 130–131. It was rejected.
The final rule preserves frames 129–141 instead of forcing a choice through
that ambiguous interval. It sacrifices some possible recovery to avoid those
observed losses.

The first, stricter mutual-swap-only prototype made no changes on any clip.
Both that null result and the rejected early-boundary result are retained.
The policy was developed on these clips, so an independent evaluation is
required before deciding whether to integrate it into the main pipeline.

## Verification and limitations

- Nineteen focused tests pass: exact disabled pass-through; supported mutual
  swaps and lost-partner repair; unchanged geometry and evidence; ambiguous
  boundaries, missing evidence, same-team cases, returning partners, later
  contradiction, competing partners/continuations including unknown teams,
  and a leading vote that does not support the confirmed team.
- The replay checks all three clips against the same frozen reference,
  reproduces previous disabled metrics, and records gained and lost correct
  observations separately. Source files, detector caches and team-read input
  hashes are preserved; output IDs are checked for collisions.
- All computation uses CPU, restricted to two cores at low priority. CUDA
  initialization remains false. No detector inference, SAM2 pass, package
  update or production job restart is performed. No speed claim is made.
- This uses the qualified colour branch, not the full SigLIP fallback or
  goalkeeper observations. Team evidence can still be wrong. It cannot resolve
  same-team swaps, absent detections, arbitrary crowd routing or global player
  identification.
- The every-frame detector caches cover 948 frames in three short clips and
  discard detections below confidence 0.30. These are limited, partly
  model-assisted reference annotations, not a new independently labeled
  full-match benchmark.

## Files on Spark

Repository: `/home/valentinweyer/projects/handball-computer-vision`

Worktree: `.codex/worktrees/arc-team-repair`

Code and reproduction commands: `experiments/arc_team_repair/README.md`

Final artifacts: `runs/arc-team-repair-final-2026-10-21/`, including the frozen
source snapshot, protocol and hashes, read observations, repaired MOT files,
event audits, per-observation matches, metrics and completion evidence.

Earlier development runs: `runs/arc-team-repair-2026-10-21/` (strict null),
`runs/arc-team-repair-v2-2026-10-21/` (rejected early boundary), and
`runs/arc-team-repair-v3-2026-10-21/` (conservative boundary).

Before/after contact sheet: `runs/arc-team-repair-final-2026-10-21/han_before_after.jpg`.

Next useful validation: freeze this rule and run it on separate longer overlap
sequences, measuring newly correct and newly wrong observations separately.
Keep the production default unchanged until that evidence exists.
