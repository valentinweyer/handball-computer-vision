# Ten-minute human review of ARC repair

This driver tests the frozen repair from `719ab8c` on an existing development
video without claiming tracking ground truth. It does not tune thresholds,
change production files, or score agreement with SAM2 as accuracy.

The queued run uses `BAL_THW_66m00_10min`, explicitly marked `role=dev` in
`runs/smart-reid-shadow/heldout_windows.json`. This is detector-training footage,
not a held-out detector test, and has not been used to develop this ARC repair.

The launcher waits for both current 76-minute Balingen II–Erlangen and
Würzburg–Regensburg processing jobs to finish tracking and bundle creation.
It also refuses to start while a pipeline inference process remains active.
The launch lock and `launched.json` prevent duplicate starts or automatic retries.

Stages, in separate processes:

1. Original RF-DETR expanded-v1 export, GPU, **every frame**, threshold 0.30,
   private cache. Existing stride-five caches do not qualify.
2. Official sports OSNet features and ARC association on CPU. Existing local
   confidence configuration (0.50/0.50), unchanged repair config and team-read
   policy, and no native offline appearance-only merger or interpolation.
   Both arms use the same raw track output and independent qualified team reads.
3. Side-by-side H.264 clips covering every accepted correction and up to twelve
   randomly chosen unchanged overlap controls. Sampling seed is 20261022.
   Controls exclude changes affecting their focal players, and do not overlap
   one another. Original-frame coordinates and all non-ID MOT fields are checked.

The review page mixes correction/control cases without labeling their type.
It accepts B-better, A-better, both-correct, both-wrong, uncertain, and bench or
off-court verdicts. Browser-local answers can be exported as JSON. A short event
clip cannot prove every later corrected frame is right; the affected intervals
and full MOT outputs remain available for deeper inspection. Numeric accuracy,
precision and regression claims require actual human review.

The deployment has a Git snapshot of the required modules, the recorded repair
hash from `719ab8c`, input hashes, a copied team model and a frozen protocol.
Changing the worktree afterward does not change that queued source snapshot.
The only GPU work is the detector after the dependency gate opens; CPU stages
use two cores with low scheduling and I/O priority. The CPU detector smoke test
was exploratory only; its modified device graph is not used in this experiment.

Prepared job on Spark:
`/home/valentinweyer/projects/handball-computer-vision/runs/arc-tenmin-review-2026-10-22`

Idempotent launcher (the heartbeat invokes this only after preparation):

```bash
cd /home/valentinweyer/projects/handball-computer-vision
.venv/bin/python runs/arc-tenmin-review-2026-10-22/runner.py \
  --root runs/arc-tenmin-review-2026-10-22
```

Adding `--check` reports the dependency state without launching. The review
package is `review/index.html`, its MP4 files and `review/manifest.json`.
`status.json`, stage logs and `complete.json` distinguish queued, running,
failed and review-ready states. A failure preserves completed artifacts and
requires deliberate investigation; do not restart automatically.
