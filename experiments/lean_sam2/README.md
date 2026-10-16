# Lean-SAM2 memory-only: no speed-up on GB10 (2026-10-16)

[Lean-SAM2](https://github.com/DeawhaleQwQ/Lean-SAM2) at `8d1cf26`, TAMP + TCIM
with the frozen release values, TARR routing off (so no bypass weights; same
SAM2.1-Large checkpoint as production). Plugged into the production driver via
`predictor_factory`; the fork ships its own `sam2` package, so it runs in its
own process (`adapter.bootstrap()`).

## Result

Han, 199 frames, tracking only, same harness and session
(`experiments/cutie/run_comparison.py --backend lean_sam2`), runs in
`runs/lean_sam2_20261016/`:

| backend | loop fps | vs production |
|---|---:|---:|
| production SAM2 | 2.35 | 1.00x |
| Lean-SAM2, all mechanisms off | 0.69 | 0.29x |
| Lean-SAM2 memory, as released | 1.33 | 0.57x |
| Lean-SAM2 memory, `stats_from_last_layer_only` | 2.25 | 0.96x |

**Why.** The fork requests attention statistics from every memory-attention
layer on every frame, which forces an explicit `softmax(QK^T)V` in place of
fused SDPA. Its reported speed-ups are against that slower baseline. Only the
last layer's statistics are read, so the adapter puts layers 0-2 back on SDPA
(pruning decisions unchanged). What remains, synchronised per-module time over
Han frames 21-80 (`ms/frame`):

| | production | Lean patched |
|---|---:|---:|
| memory-attention layer 0-2, each | 60 | 37 |
| memory-attention layer 3 (explicit, for the statistics) | 60 | 142 |
| image encoder | 93 | 93 |
| frame | 435 | 459 |

Pruning a memory layer saves ~40%, not 75%: self-attention over the 4096 image
tokens per player, projections and the FFN are not pruned. Even with free
statistics the ceiling is ~4 x 37 ms against 4 x 60 ms, i.e. ~1.25x on the
whole frame; computing the last layer's statistics cheaply (fused attention
with log-sum-exp plus a key-chunked usage pass) would land nearer 1.1-1.2x.

**Correctness.** With every mechanism off the fork reproduces production's IDs
on every frame; boxes differ by explicit-vs-SDPA BF16 numerics (median 1 px,
3 of 2664 above 20 px). Memory mode also keeps identical IDs on Han; mask
outlines move more (99th percentile 103 px).

## Verdict

Not worth pursuing for speed: the best case is ~1.2x, far from what the
tracker needs. Kept as an experiment. `review_diff.py` (side-by-side clips
wherever a candidate and SAM2 disagree on identity) is generic and reusable
for the next candidate.
