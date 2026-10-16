"""Trained SAM2 decoder/memory-encoder graph exactness and latency microtest."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=40)
    args = p.parse_args()
    assets = Path('/home/valentinweyer/projects/handball-computer-vision')
    os.environ.setdefault('TRITON_PTXAS_PATH', str(assets / '.venv/lib/python3.11/site-packages/torch/bin/ptxas'))
    sys.path.insert(0, str(assets / 'sam2-upstream'))
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2.modeling.sam.mask_decoder import MaskDecoder
    from sam2.modeling.sam.transformer import TwoWayTransformer
    from experiments.sam2_profile.patches.cuda_small_graphs import SmallModuleGraphs, _map

    torch.manual_seed(0)
    config = OmegaConf.load(assets / 'sam2-upstream/sam2/configs/sam2.1/sam2.1_hiera_l.yaml')
    encoder = instantiate(config.model.memory_encoder).eval()
    decoder = MaskDecoder(transformer_dim=256,
                          transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
                          num_multimask_outputs=3, iou_head_depth=3, iou_head_hidden_dim=256,
                          use_high_res_features=True, iou_prediction_use_sigmoid=True,
                          dynamic_multimask_via_stability=True, dynamic_multimask_stability_delta=0.05,
                          dynamic_multimask_stability_thresh=0.98, pred_obj_scores=True,
                          pred_obj_scores_mlp=True, use_multimask_token_for_obj_ptr=True).eval()
    weights = torch.load(assets / 'segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt',
                         map_location='cpu', mmap=True, weights_only=True)['model']
    for name, module in [('memory_encoder', encoder), ('sam_mask_decoder', decoder)]:
        prefix = name + '.'
        module.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True)
        module.cuda()
    del weights

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device='cuda', dtype=dtype)

    def encoder_values():
        return dict(pix_feat=rand(1, 256, 64, 64), masks=rand(1, 1, 1024, 1024, dtype=torch.float32),
                    skip_mask_sigmoid=True)

    def decoder_values(multimask, points):
        image = rand(4096, 1, 256).permute(1, 2, 0).view(1, 256, 64, 64)
        pos = rand(4096, 1, 256).permute(1, 2, 0).view(1, 256, 64, 64)
        return dict(image_embeddings=image, image_pe=pos,
                    sparse_prompt_embeddings=rand(1, points, 256, dtype=torch.float32),
                    dense_prompt_embeddings=rand(1, 256, 1, 1, dtype=torch.float32).expand(1, 256, 64, 64),
                    multimask_output=multimask, repeat_image=False,
                    high_res_features=[rand(1, 32, 256, 256), rand(1, 64, 128, 128)])

    def flat(value):
        tensors = []
        _map(value, lambda t: tensors.append(t))
        return tensors

    def equal(a, b):
        x, y = flat(a), flat(b)
        return len(x) == len(y) and all(torch.equal(u, v) for u, v in zip(x, y))

    def measure(fn):
        for _ in range(4):
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

    wrappers = {name: SmallModuleGraphs(module, min_calls=1, max_entries=2)
                for name, module in [('memory_encoder', encoder), ('sam_mask_decoder', decoder)]}
    cases = [('memory_encoder', encoder_values),
             ('sam_mask_decoder', lambda: decoder_values(True, 2)),
             ('sam_mask_decoder', lambda: decoder_values(False, 3))]
    rows = []
    for name, factory in cases:
        wrapper = wrappers[name]
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            values = factory()
            ref = wrapper.original(**values)
            torch.cuda.synchronize()
            began = time.perf_counter()
            output = wrapper(**values)
            torch.cuda.synchronize()
            capture_ms = (time.perf_counter() - began) * 1000
            assert equal(ref, output), f'{name} first graph output differs'
            retained = _map(output, lambda t: t.clone())
            changed = factory()
            assert equal(wrapper.original(**changed), wrapper(**changed)), f'{name} inputs were stale'
            assert equal(output, retained), f'{name} retained output overwritten'
            eager = measure(lambda: wrapper.original(**values))
            graph = measure(lambda: wrapper(**values))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            assert equal(wrapper.original(**values), wrapper(**values)), f'{name} autocast lifetime failed'
        rows.append({'module': name, 'multimask': values.get('multimask_output'),
                     'equal': True, 'dynamic_inputs_equal': True, 'retained_output_equal': True,
                     'autocast_boundary_equal': True, 'capture_wall_ms': capture_ms,
                     'eager': eager, 'graph': graph, 'wall_speedup': eager['wall_ms'] / graph['wall_ms']})
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        third = decoder_values(True, 4)
        assert equal(wrappers['sam_mask_decoder'].original(**third), wrappers['sam_mask_decoder'](**third))
        assert len(wrappers['sam_mask_decoder'].entries) == 2
    result = {'torch': torch.__version__, 'device': torch.cuda.get_device_name(), 'results': rows,
              'stats': {name: wrapper.stats() for name, wrapper in wrappers.items()},
              'bounded_cache_fallback_equal': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
