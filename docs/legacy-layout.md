# Legacy layout retained during migration

No original code, media, checkpoint, output, cache, or upstream checkout was
deleted or moved in the first restructuring pass.

The following remain intentionally:

- `notebooks/*.py`: original working scripts and modules.
- `outputs/`: old render outputs, fitted models, caches, and annotations.
- `source/`: old source media, frame caches, mask caches, and tracker runs.
- `data/raw/`: canonical raw-video location after the root-video migration.
- `runs/legacy/`: retained diagnostic encodes moved out of the repository root.
- `onnxruntime/`, `sam3/`, and `segment-anything-2-real-time/`: nested
  upstream repositories.
- `sam3-checkpoints/`: SAM3.1 weights retained for jersey-number experiments.

The canonical copies are now under `src/`, `scripts/`, `experiments/`, and
`data/annotations/`. Only those copies should receive new structural work.

Deletion should happen in a later, explicit pass after:

1. the package tests pass;
2. key scripts run from the new paths;
3. annotation copies are compared by checksum;
4. upstream repositories are checked for local modifications;
5. duplicate checkpoints are compared by SHA-256;
6. the user approves the exact deletion list.
