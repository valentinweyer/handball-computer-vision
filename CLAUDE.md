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
`data/annotations/`. Older `outputs/` and `source/` paths still exist and still
work as migration fallbacks. The 29 legacy `notebooks/*.py` module copies were
deleted on 2026-09-09 after they drifted from the modules that replaced them;
`notebooks/` now holds only the `.ipynb` files and their assets (`fonts/`,
`.env`, `models/`, dataset directories). New code imports from `handball_cv`;
new commands run as modules:

```bash
python -m scripts.render_mcbyte_team_correction --help
```

## Hard constraints

- Use the local RF-DETR/Roboflow detector.
- SAM2 with periodic detector reprompting is the tracker default as of
  2026-09-08, on the strength of three clips scored against ground truth
  (`docs/tracking-evaluation.md` §8: 95.0/89.2/98.9% correct against
  81.5/86.7/95.9% for the best box tracker on each; FelixClaar was 93.8% when
  §8.2 was written and measured 95.0% after the lifecycle work, see §8.10). MCByte remains supported
  and is roughly 10x faster (~0.1s/frame against ~1s); pick it explicitly when
  cost matters. Do not switch to ByteTrack.
- The jersey reader default is `parseq` (the original baudm/parseq checkpoint),
  measured at 0.858 accuracy on the 323 labelled 1080p crops against docTR
  parseq's 0.622, Qwen's 0.628 and EasyOCR's 0.365. It needs an external
  checkout and a local checkpoint and fails loudly without them.
- Re-ID appearance features default to `prtreid`. The team model's own features
  cannot separate teammates (0.35 rank-1 against a 0.19 chance floor, versus
  0.55), and team classification keeps its own features either way.
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
uv run --no-sync pytest -q tests
```

`--no-sync` is required, not cosmetic: there is no `uv.lock`, and `torch` is not
in `[project].dependencies` — it comes from the aarch64/GB10 CUDA install
recorded in `requirements.txt`. A default `uv run` would resolve from
`pyproject.toml` alone and can uninstall it.

Do not run bare `pytest` from the repository root — vendored `onnxruntime` and
other upstream trees contain unrelated test entry points that break global
collection.

## Repository-state warning

The working tree contains many pre-existing modified and untracked experiment
files. They belong to the ongoing project. Do not run destructive cleanup,
reset, or checkout commands. Work only on the requested files and preserve
unrelated changes.
