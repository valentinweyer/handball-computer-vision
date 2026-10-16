# SAM2 on GB10: bandwidth or compute? And a bit-identical fused RoPE (2026-10-16)

Measured on the GB10 with no other GPU work running, SAM2.1-L with BF16
autocast, eager, `CHECK_EVERY=10`, cached RF-DETR detections. Scripts are in
`experiments/sam2_profile/`, outputs in `runs/sam2_gpu_bottleneck/` (gitignored).
This extends [inference-speed-verdict.md](inference-speed-verdict.md) §0 and
corrects one of its conclusions (fused RoPE; see below).

## Answer

**Neither alone. About half of the GPU time is memory-bound, about a quarter is
compute-bound attention, and a fifth is GEMMs that reach neither roof.**
Han-Ber4, 14 players a frame, the 22 steady-state frames 8-32 with the memory
bank full and the post-checkpoint frames excluded:

| | GPU ms/frame | Share of busy | Measured rate | Limited by |
|---|---:|---:|---:|---|
| Memory-streaming kernels: casts, add/mul, layer norm, GELU, copies | 169.6 | 49% | ≥53.6 GB/frame compulsory, 316 GB/s effective | Memory traffic; no arithmetic |
| Flash attention | 98.4 | 28% | 80.5 TFLOPS, 85% of the measured attention roof | Compute |
| cuBLAS GEMM | 75.1 | 22% | 33.7 TFLOPS, 53% of the measured GEMM roof; 210 FLOP/byte | Neither roof; kernel choice |
| Convolution | 3.8 | 1% | | |
| **GPU idle** | **41.4** | 11% of the 388 ms frame | | Host work, launches, syncs |

The memory class's 316 GB/s exceeds the DRAM ceiling, so part of its traffic is
served from L2. Without hardware counters, DRAM-bound and L2-bound cannot be
separated; "memory traffic" is the accurate claim.

By module (GPU ms/frame):

| Module | Total | Attention | GEMM | Conv | Memory |
|---|---:|---:|---:|---:|---:|
| `memory_attention` cross-attention | 140.5 | 76.0 | 10.9 | 0 | 53.5 |
| `image_encoder` | 88.4 | 7.1 | 36.8 | 0 | 44.5 |
| `memory_attention` FFN / norms / residuals | 32.9 | 0 | 11.3 | 0 | 21.6 |
| `memory_encoder` | 28.4 | 0 | 6.8 | 3.0 | 18.7 |
| `memory_attention` self-attention | 27.0 | 14.2 | 4.6 | 0 | 8.2 |
| `sam_mask_decoder` | 18.5 | 1.0 | 4.4 | 0.9 | 12.3 |
| predictor glue | 7.7 | 0 | 0 | 0 | 7.7 |

**The largest memory sink was RoPE on the memory-bank keys.** For every
player and every memory-attention layer, upstream `apply_rotary_enc` casts a
28,672 x 256 BF16 key tensor to FP32, multiplies it as complex64 and casts it
back: about 13 GB/frame of casts and elementwise traffic in cross-attention
alone. Weight casts are negligible (autocast caches them); the traffic is
activations. This is also why the LLM-decode toolbox only partly transfers:
weight quantization and speculative decoding attack weight reads and
autoregression, neither of which dominates here.

## Measured ceilings (`ceilings.py`, `ceilings.json`)

| | Measured |
|---|---:|
| cuBLAS BF16 GEMM, 4096³-8192³ | 62.6-64.4 TFLOPS |
| GEMM [28736 x 64] x [64 x 256] (cross-attn `k_proj` shape) | 13.2 TFLOPS |
| Flash attention, q 4096 x k 28736 x d256 (the cross-attn shape) | 88.1 TFLOPS |
| Flash attention, 8192² x d64 x 16 heads | 94.8 TFLOPS |
| DRAM copy / read | 223 / 240 GB/s (spec 273) |

cuBLAS selects Ampere-generation `cutlass_80` tensor-op and `wmma` kernels on
this sm_121 GPU, many with 16x16 or 32x32 tiles for SAM2's shapes. The ridge
point is ~270-400 FLOP/byte.

## Method, and what it cannot see

`ncu` hardware counters are blocked for non-admin users
(`RmProfilingAdminOnly=1`, `ERR_NVGPUCTRPERM`), and `torch.profiler` records no
CUDA kernels on this setup (as §0 of the verdict doc also found).
**`nsys profile -t cuda,nvtx` does work as a normal user**, and it is what
supplies kernel times here. Three measurements are joined:

1. **Ceilings** from microbenchmarks (`ceilings.py`).
2. **FLOPs and compulsory bytes per aten op** (`roofline.py --mode roofline`):
   a `TorchDispatchMode` records each op's FLOPs (PyTorch's flop-counter
   formulas) and the unique memory spans of its inputs and outputs, attributed
   to the owning module via forward hooks. Composite ops (`linear`, `sdpa`,
   `to`, `reshape`, ...) must be decomposed on an allowlist: undecomposed they
   carry no FLOP formula, and decomposing everything swaps in Python reference
   math that production never runs (`upsample_bilinear2d.vec` became
   `_unsafe_index` arithmetic). This run's per-op event times are inflated by
   the mode's CPU overhead (558 or 459 ms against ~380 ms of real GPU time),
   so only its FLOPs and bytes are used.
3. **True kernel time** from a clean run under nsys (`roofline.py --mode
   plain`) with an NVTX range per module and per frame; `analyze_nsys.py`
   attributes each kernel to the innermost module open at launch. The NVTX
   hooks cost nothing measurable: steady frames ran at ~405 ms, the
   documented propagation time.

Compulsory bytes are a lower bound on traffic (cache re-reads are not counted,
L2 hits are counted as DRAM). One clip, one session.

## Fused RoPE: bit-identical after all

[inference-speed-verdict.md](inference-speed-verdict.md) §0 concluded that
RoPE's FP32 copies "could be fused, but a fused kernel would round
differently". That holds for the obvious fusions and not in general:

- `torch.compile` of upstream `apply_rotary_enc`: identical, but 1.02 -> 0.98 ms
  (Inductor does not fuse the complex ops; the `repeat` still materialises).
- `torch.compile` of a real-arithmetic rewrite: 0.15 ms, but 56 of 7.3M key
  values differ by one BF16 ulp (FMA contraction).
- A Triton kernel with explicit FMAs, all nine orderings tried: **bit-identical
  on both q and k only with `real = fma(a, c, -(b*d))`,
  `imag = fma(b, c, a*d)`**, which is what eager complex64 multiply compiles
  to on this stack. 1.03 -> 0.147 ms per cross-attention call (7x).

`experiments/sam2_profile/fused_rope.py` installs it on top of production's
`install_rope_repeat_cache`, falling back to the previous function for
anything that is not contiguous BF16 on CUDA with a frequency table that tiles
the sequence. It reads BF16 once, rotates in FP32 registers and writes BF16
once, indexing the frequency table modulo its length instead of repeating it.

End to end on Han-Ber4, all 198 frames, comparing ids, boxes and a SHA-256 of
every frame's packed masks:

| Harness | Baseline | Fused RoPE | Baseline again | Output |
|---|---:|---:|---:|---|
| `roofline.py --mode rope-ab`, in one process, mean ms/frame | 425.3 | **380.5** | 429.4 | identical on 198/198 frames |
| `ab.py`, one process per config, loop wall | 85.0 s | **75.7 s** (1.12x, 2.33 -> 2.61 fps) | | identical on 198/198 frames |

**Status: experiment only.** Not in `sam2_driver.py`; not yet checked on
FelixClaar or BHC-FAG; and it needs `TRITON_PTXAS_PATH` pointing at a ptxas
that knows sm_121 (Triton's bundled ptxas does not; see
[sam2-speed-research.md](sam2-speed-research.md), Stage A). A production
version must fall back to the current path when Triton is unavailable.

## Remaining candidates

Estimates against the 388 ms steady frame, before fused RoPE; none measured:

| Candidate | Attacks | Estimate | Output |
|---|---|---:|---|
| CUDA graphs per player step | 41 ms idle plus launch gaps; ~19k ops/frame, ~4.7k under 20 µs | 25-40 ms | Same kernels: identical in principle; one graph per memory-bank shape; must clone outputs |
| In-place RoPE write | the `k[:, :, :n] = ...` copy, ~1.6 GB/frame | ~5 ms | identical |
| Image encoder elementwise fusion | 44.5 ms of memory traffic | 20-25 ms | changes output (compiled encoder) |
| GEMM kernel selection | 75 ms at 34 TFLOPS | up to 20-30 ms | changes output unless the same kernel is chosen |
| Caching memory-bank projections ("KV cache") | `v_proj` exactly cacheable; `k_proj(memory + pos)` is not, because the temporal encoding shifts each frame | ~10 ms after fused RoPE | values: identical only if per-chunk GEMMs match; keys: no |

Already measured and dropped in [inference-speed-verdict.md](inference-speed-verdict.md) §0:

- **batching objects**: slower on GB10, 212 -> 242 ms, and not identical;
- **other attention backends**: memory-efficient 1.9x and math 9.2x slower;
- **skipping the repeated boundary frame**: changes output.

The post-checkpoint frames (~790 ms against ~400, one in ten frames, ~40 ms/frame
averaged) remain the largest single target outside the per-frame kernels.

**For a bigger GPU:** the memory half should scale with bandwidth (6.6x spec on
an RTX 5090 or RTX PRO 6000, less for small kernels). Attention scales only
with BF16 tensor throughput at FP32 accumulation, where this GPU already reaches
80 TFLOPS. Whether GeForce Blackwell halves that rate at FP32 accumulation was
not confirmed against NVIDIA documentation. Idle time and host work do not
shrink with the GPU.

## Reproduce

From the worktree root, with the project venv's python and nothing else on the GPU:

```bash
python -m experiments.sam2_profile.ceilings > runs/sam2_gpu_bottleneck/ceilings.json
python -m experiments.sam2_profile.roofline --mode roofline --output runs/sam2_gpu_bottleneck/roofline
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o runs/sam2_gpu_bottleneck/plain_nsys \
    python -m experiments.sam2_profile.roofline --mode plain --output runs/sam2_gpu_bottleneck/plain
nsys export -t sqlite -o runs/sam2_gpu_bottleneck/plain_nsys.sqlite runs/sam2_gpu_bottleneck/plain_nsys.nsys-rep
python -m experiments.sam2_profile.analyze_nsys runs/sam2_gpu_bottleneck/plain_nsys.sqlite \
    runs/sam2_gpu_bottleneck/roofline/ops.jsonl
python -m experiments.sam2_profile.ab run --case han --output base.json
python -m experiments.sam2_profile.ab run --case han --patch experiments.sam2_profile.fused_rope:install --output rope.json
python -m experiments.sam2_profile.ab compare base.json rope.json
```

`ab.py` is the general A/B harness: patches are `module:function` hooks called
with the predictor, each configuration runs in its own process, and `compare`
reports per-frame identity of ids, boxes and mask hashes with wall times. The
cases are `han`, `felix` and `bhc`. `diagnose_gpu.py` is an earlier
module-level CUDA-event probe from the same investigation.
