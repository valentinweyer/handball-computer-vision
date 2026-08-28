# Handball CV: working rules

Project detail — architecture, calibration, measurements, experiment results,
rejected approaches, data assets, file inventory, result artifacts, and the
agreed next implementation step — lives in
`docs/team-classification-handoff.md`. **Read it before changing the detection,
tracking, team-classification, or identity code.** It is not auto-loaded; open
it when the task touches the pipeline.

Also: `docs/architecture.md` (dependency rules), `docs/legacy-layout.md`,
`docs/tracking-evaluation.md` (per-frame tracker measurements; supersedes older
tracker claims), `docs/identity-decoupling.md`,
`docs/overlap-mask-experiment.md`.

## Layout

Reusable code is `src/handball_cv/`; CLI diagnostics are `scripts/`; rejected or
non-default approaches are `experiments/`; protected label copies are
`data/annotations/`. Older `notebooks/*.py`, `outputs/`, and `source/` paths
still exist and still work as migration fallbacks. New code imports from
`handball_cv`; new commands run as modules:

```bash
python -m scripts.render_mcbyte_team_correction --help
```

## Hard constraints

- Use the local RF-DETR/Roboflow detector.
- MCByte is the production tracker default. Do not switch to ByteTrack. Tracker
  choice is under active reconsideration — SAM2 with periodic detector
  reprompting measured best on all three clips tested — so consult
  `docs/tracking-evaluation.md` §8 before changing the default.
- Raw team classification stays frame-local. A tracker must not supply or freeze
  a raw team prediction.
- Team labels attached to identities must stay reversible.
- Abstaining beats injecting a confident wrong observation.
- No circular coupling: a team error must not force a tracking error, and a
  tracking error must not automatically become a team error.
- Do not enable team-gated association (`team_aware_tracker.py`) by default.
- Teams are discovered anonymously per video. Do not build a global supervised
  team catalogue, and do not start a large labeling project.

## Before editing pipeline code

These areas carry invariants that break silently. Read the named handoff section
first:

- Team evidence, decay, or the re-ID veto in `identity_manager.py` →
  "Decoupling tracker error from team error".
- Mask selection or masked features → "Guarded mask construction". Never select
  a target mask by tracker ID alone.
- Crop geometry or thresholds in `team_model.py` → "Per-video team-model
  calibration".
- Whether something is actually wired into the runtime → "Current
  implementation status". Several modules are implemented but not connected.

## Commands

```bash
conda run -n NewEnv pytest -q tests
```

Do not run bare `pytest` from the repository root — vendored `onnxruntime` and
other upstream trees contain unrelated test entry points that break global
collection.

## Repository-state warning

The working tree contains many pre-existing modified and untracked experiment
files. They belong to the ongoing project. Do not run destructive cleanup,
reset, or checkout commands. Work only on the requested files and preserve
unrelated changes.
