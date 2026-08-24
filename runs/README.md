# Run artifacts

Generated experiment results belong here and are ignored by Git. New runs should
use a stable layout such as:

```text
runs/<experiment>/<video-id>/
├── run.json
├── metrics.json
├── overlay.mp4
└── diagnostics/
```

`run.json` should record the source-video fingerprint, detector/model versions,
configuration, command, and cache paths. Reusable detections and masks belong in
`data/cache/`, not in the run directory.

The existing `outputs/` and `source/.runs/` trees are preserved as legacy runs.
