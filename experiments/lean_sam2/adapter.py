"""Lean-SAM2 (DeawhaleQwQ/Lean-SAM2) as a drop-in SAM2 predictor for drive_sam2.

Lean-SAM2 is a fork of SAM2 that keeps the SAM2.1 weights and makes memory
attention cheaper:

* TAMP (Target-Anchored Memory Pruning) keeps a quarter of the memory tokens.
* TCIM (Temporal Condensation with Insurance Memory) folds older memories into
  one EMA state and keeps up to three reliable "insurance" frames.

Its third module, TARR, routes background image windows through a trained
bypass and needs extra weights. It is left off here ("memory-only"), so the
checkpoint is the same SAM2.1-Large file production uses.

The fork ships its own `sam2` package, so `bootstrap()` must put the checkout
first on sys.path before anything imports `sam2`, and a run cannot mix it with
the production upstream checkout in one process.

Methods:
  memory  the frozen release values for TAMP and TCIM (configs/lean_sam2/lean_sam2.yaml)
  off     every Lean-SAM2 mechanism disabled; must reproduce production SAM2
          exactly, which is the check that the fork and this adapter change
          nothing else
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEAN = ROOT / '.local/lean-sam2'
REVISION = '8d1cf2618519d291905d22fe859b8302476d7f50'
METHOD_FILE = LEAN / 'configs/lean_sam2/lean_sam2.yaml'
# Release YAML key -> SAM2Base argument, where the names differ.
RENAMED = {'enable_anchor': 'anchor_enabled', 'enable_insurance': 'enable_insurance_bank'}
# Release keys that are not model arguments: the mode label, the CFM runtime
# profile (CFM itself is off), and everything belonging to TARR routing.
NOT_MODEL = {'mode', 'cfm_runtime_profile', 'use_swr'}
OFF = {'mem_keep_ratio': 1.0, 'anchor_enabled': False, 'enable_insurance_bank': False,
       'enable_diversity_bank': False, 'enable_cross_frame_merge': False,
       'enable_ema_history_bank': False}


def bootstrap():
    """Make `import sam2` resolve to the fork, even after a harness has put the
    production upstream checkout on the path."""
    sys.path[:] = [p for p in sys.path if Path(p).name != 'sam2-upstream']
    for path in [ROOT, ROOT / 'src']:
        if str(path) not in sys.path:
            sys.path.append(str(path))
    sys.path.insert(0, str(LEAN))


def revision():
    head = subprocess.check_output(['git', '-C', str(LEAN), 'rev-parse', 'HEAD'], text=True).strip()
    if head != REVISION:
        raise RuntimeError(f'{LEAN} is at {head}, expected the audited {REVISION}')
    return head


def method_values(method):
    if method == 'off':
        return dict(OFF)
    if method != 'memory':
        raise ValueError(f"method must be 'memory' or 'off', got {method!r}")
    import yaml
    frozen = yaml.safe_load(METHOD_FILE.read_text())
    return {RENAMED.get(key, key): value for key, value in frozen.items()
            if key not in NOT_MODEL and not key.startswith('swr_')}


def build_predictor(checkpoint, method):
    """-> (predictor, model arguments applied). Call bootstrap() first."""
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(LEAN):
        raise RuntimeError(f'sam2 resolved to {sam2.__file__}; call bootstrap() before importing it')
    from sam2.build_sam import build_sam2_video_predictor
    from handball_cv.tracking.sam2_driver import SAM2_CONFIG, install_rope_repeat_cache
    values = method_values(method)
    overrides = [f'++model.{key}={str(value).lower() if isinstance(value, bool) else value}'
                 for key, value in values.items()]
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(checkpoint), hydra_overrides_extra=overrides,
                                           use_swr=False)
    for key, value in values.items():
        if getattr(predictor, key) != value:
            raise RuntimeError(f'{key}: asked for {value!r}, model has {getattr(predictor, key)!r}')
    # The same two settings production applies to its own predictor.
    predictor.fill_hole_area = 0
    install_rope_repeat_cache()
    stats_from_last_layer_only(predictor.memory_attention)
    return predictor, values


def stats_from_last_layer_only(memory_attention):
    """Let all but the last memory-attention layer use fused SDPA.

    The fork asks every layer for attention statistics on every frame, which
    forces its explicit softmax(QK^T)V path (the full 4096 x memory-tokens
    matrix per player) in all four layers. MemoryAttention.forward keeps only
    the last layer's statistics (`attn_stat = layer_usage` overwrites), so the
    other three compute them for nothing. On Han this made the fork with every
    mechanism off 3.4x slower than production SAM2 (0.69 vs 2.36 fps).

    The pruning decisions are unchanged: they read the same last-layer
    statistics. Layers 0-2 move from explicit attention to SDPA, the same
    arithmetic production uses.
    """
    for layer in memory_attention.layers[:-1]:
        forward = layer.forward

        def without_stats(*args, return_attn_stat=False, _forward=forward, **kwargs):
            out = _forward(*args, return_attn_stat=False, **kwargs)
            return (out, None) if return_attn_stat else out

        layer.forward = without_stats
