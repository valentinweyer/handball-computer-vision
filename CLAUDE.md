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
  81.5/86.7/95.9% for the best box tracker on each). Two caveats on those
  figures. FelixClaar was 93.8% when §8.2 was written and measured 95.0% after
  the lifecycle work (§8.10). Han-Ber4's 89.2% divides by an all-reference
  denominator that includes bench and referee ids the numerator cannot draw
  from; on the same players it is 99.91% (§8.11). Every tracker in that
  comparison shares the denominator, so 89.2-against-86.7 is like-for-like --
  but do not quote 89.2 as this tracker's accuracy. MCByte remains supported
  and is roughly 5x faster (~0.1s/frame against ~0.46s; SAM2 was ~1s until the
  CPU mask work in `docs/sam2-speed-research.md`, and MCByte does not share that
  code path); pick it explicitly when cost matters. Do not switch to ByteTrack.
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

**~20 GB of untracked, gitignored artifacts sit in the working tree, and most
of them cost GPU hours to rebuild.** They are ignored precisely so they survive;
ignored does not mean disposable here:

- `data/cache/frames/` (9.6 GB, 15 clips) -- JPEG frames the SAM2 path reads
  instead of decoding the video. Both the tracking pass and `--redraw` read
  these, and a redraw that read the mp4 instead would not reproduce the render.
- `runs/` (7.6 GB) -- renders, run summaries, and the `*_reads.json` and
  `*_geometry.npz` caches. The reads cache saves a full reader pass; the
  geometry cache saves a full tracking pass (~22 min for a 1499-frame clip
  against ~80 s to redraw from it).
- `outputs/` (2.5 GB) -- detection and team-model caches, plus the migration
  fallbacks.

So: do not run `git clean`, and do not run destructive cleanup, reset, or
checkout commands. Work only on the requested files and preserve unrelated
changes. If a cache looks stale, regenerate it deliberately rather than
deleting the tree it lives in.

Earlier versions of this file warned that the tree was full of modified and
untracked *experiment* files. That is no longer true -- the tree has been clean
since the 2026-09-11 merge -- but the artifact directories above are the real
hazard and always were.
