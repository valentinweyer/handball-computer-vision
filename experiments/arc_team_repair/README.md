# ARC team-confirmed overlap repair

An offline, default-off experiment on `experiment/arc-team-repair`. It consumes
the saved every-frame ARC-SORT CPU experiment from `def6081`; it does not replace
SAM2, add a production pipeline option, or change global player identities.

`repair.py` supports two narrow cases:

1. Two live tracks have opposite reliable teams before contact, mutually
   reversed teams afterward, and a uniquely supported geometric switch. Swap
   the IDs retrospectively from that boundary.
2. One track ends during contact and the survivor changes to its team. Require
   a short spatial bridge, independently qualified team evidence and a unique
   lost partner. Relabel the survivor's tail to the lost ID starting at the
   first qualified clear observation, confirmed by subsequent votes. Keep the
   ambiguous interval unchanged.

The second case can be considered in a crowd, but competing spatial routes
veto the repair even if their team observations are missing. Same-team contact,
weak observations, later team contradictions and returning lost tracks abstain.
Only one correction involving any given raw track is allowed. Each correction
is recorded in an audit. Coordinates, row order, detection confidence and raw
team observations remain unchanged. Source MOT files are never overwritten.

The flag is `--team-overlap-repair`. Without it, output is byte-identical to the
input MOT file and no team model is loaded. The native ARC appearance-only
offline track merger is not used. Native gap interpolation is scored separately.

## Reproduce on Spark

Requires the existing primary venv, detector caches, per-video team models,
reference annotations, and the completed `runs/arc-sort-cpu-2026-10-21` artifacts.
Do not install or synchronize dependencies into that venv.

```bash
cd /home/valentinweyer/projects/handball-computer-vision/.codex/worktrees/arc-team-repair
MAIN=/home/valentinweyer/projects/handball-computer-vision
BASE=$MAIN/runs/arc-sort-cpu-2026-10-21
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH="$BASE/deps:$BASE/vendor/arcsort:src:." \
  taskset -c 0,1 nice -n 15 ionice -c 3 "$MAIN/.venv/bin/python" \
  -m experiments.arc_team_repair.run --team-overlap-repair \
  --root "$MAIN/runs/arc-team-repair-new-evaluation"
```

Use a new result directory for a changed policy. `protocol.json` locks the
configuration and source hashes before scoring. Team-read caches check model,
video, detections, source tracks and observation-policy hashes before reuse.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH=src:. taskset -c 0,1 "$MAIN/.venv/bin/python" \
  -m pytest -q experiments/arc_team_repair/test_repair.py
```

## Evidence and limits

See `docs/arc-team-repair-2026-10-21.md`. These are development clips, not a
held-out confirmation. The experiment uses only the existing per-video colour
classifier's qualified field-player observations, not the SigLIP fallback or
goalkeeper labels. The every-frame detector cache is truncated below confidence
0.3. No speed claim or production recommendation follows from these tests.
