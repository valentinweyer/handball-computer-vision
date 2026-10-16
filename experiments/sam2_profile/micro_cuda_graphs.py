"""Trained memory-attention graph identity, lifetime and timing microtest.

Run only with exclusive GPU timing access. Produces a JSON report; any identity
failure raises immediately. The full A/B loop remains the acceptance gate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    assets = Path('/home/valentinweyer/projects/handball-computer-vision')
    os.environ.setdefault('TRITON_PTXAS_PATH', str(assets / '.venv/lib/python3.11/site-packages/torch/bin/ptxas'))
    sys.path.insert(0, str(assets / 'sam2-upstream'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from experiments.sam2_profile.fused_rope import install as install_rope
    from experiments.sam2_profile.patches.cuda_graphs import MemoryAttentionGraphs
    from handball_cv.tracking.sam2_driver import install_rope_repeat_cache

    config = OmegaConf.load(assets / 'sam2-upstream/sam2/configs/sam2.1/sam2.1_hiera_l.yaml')
    torch.manual_seed(0)
    module = instantiate(config.model.memory_attention).eval()
    weights = torch.load(assets / 'segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt',
                         map_location='cpu', mmap=True, weights_only=True)['model']
    module.load_state_dict({k.removeprefix('memory_attention.'): v for k, v in weights.items()
                            if k.startswith('memory_attention.')}, strict=True)
    del weights
    module.cuda()
    install_rope_repeat_cache()
    install_rope(SimpleNamespace())
    original = module.forward
    wrapper = MemoryAttentionGraphs(module, min_calls=1, max_entries=2)

    def inputs(tokens, pointers):
        return dict(curr=[torch.randn(4096, 1, 256, device='cuda', dtype=torch.bfloat16)],
                    memory=torch.randn(tokens + pointers, 1, 64, device='cuda', dtype=torch.bfloat16),
                    curr_pos=[torch.randn(4096, 1, 256, device='cuda', dtype=torch.bfloat16)],
                    memory_pos=torch.randn(tokens + pointers, 1, 64, device='cuda', dtype=torch.bfloat16),
                    num_obj_ptr_tokens=pointers)

    def measure(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        began = time.perf_counter()
        start.record()
        for _ in range(args.repeats):
            fn()
        end.record()
        end.synchronize()
        return {'cuda_ms': start.elapsed_time(end) / args.repeats,
                'wall_ms': (time.perf_counter() - began) * 1000 / args.repeats}

    rows = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for tokens, pointers in [(28672, 64), (24576, 32)]:
            values = inputs(tokens, pointers)
            ref = original(**values)
            torch.cuda.synchronize()
            began = time.perf_counter()
            out = wrapper(**values)
            torch.cuda.synchronize()
            capture_ms = (time.perf_counter() - began) * 1000
            equal = torch.equal(ref, out)
            assert equal, f'First graph output differs for {tokens}+{pointers}'
            old = out.clone()
            changed = inputs(tokens, pointers)
            changed_ref = original(**changed)
            changed_out = wrapper(**changed)
            assert torch.equal(changed_ref, changed_out), 'Static-input update was stale'
            assert torch.equal(out, old), 'A retained graph output was overwritten'
            eager = measure(lambda: original(**values))
            graph = measure(lambda: wrapper(**values))
            rows.append({'memory_tokens': tokens, 'pointer_tokens': pointers,
                         'equal': equal, 'dynamic_inputs_equal': True, 'retained_output_equal': True,
                         'capture_wall_ms': capture_ms, 'eager': eager, 'graph': graph,
                         'wall_speedup': eager['wall_ms'] / graph['wall_ms']})
        # A third signature cannot grow the two-entry test cache.
        unseen = inputs(4096, 4)
        assert torch.equal(original(**unseen), wrapper(**unseen))
        assert len(wrapper.entries) == 2
    # Exiting driver chunks clears cached autocast parameter conversions.
    # Replay must still own every pointer used by the captured graph.
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        assert torch.equal(original(**values), wrapper(**values)), 'Autocast boundary changed graph output'
    result = {'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
              'results': rows, 'stats': wrapper.stats(), 'bounded_cache_fallback_equal': True, 'autocast_boundary_equal': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
