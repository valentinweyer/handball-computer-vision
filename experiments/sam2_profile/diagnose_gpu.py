"""Bounded production SAM2 trace and representative attention measurements.

No model/pipeline code changes. Reads cached Han-Ber4 inputs from --assets-root.
Use --mode counters under ncu to profile only one representative memory layer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-root', type=Path, default=Path('/home/valentinweyer/projects/handball-computer-vision'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['trace', 'counters'], default='trace')
    p.add_argument('--frames', type=int, default=32)
    p.add_argument('--warmup', type=int, default=12)
    p.add_argument('--sample-frame', type=int, default=25)
    args = p.parse_args()
    if not 1 <= args.sample_frame < args.frames or args.warmup < 2:
        p.error('sample-frame must be inside the clip; warmup >= 2')
    if args.output.exists():
        p.error('output must be a new directory')
    base = args.assets_root.resolve()
    for path in [base/'sam2-upstream', ROOT/'src', ROOT]:
        sys.path.insert(0, str(path))
    os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sam2.build_sam import build_sam2_video_predictor
    from handball_cv.teams.model import TeamModel
    from handball_cv.tracking.sam2_driver import drive_sam2, install_rope_repeat_cache
    from scripts.render_raw_team_classification import load_detection_cache, person_detections

    torch.manual_seed(0)
    args.output.mkdir(parents=True)
    checkpoint = base/'segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt'
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',
        ckpt_path=None, device='cpu', vos_optimized=False,
        hydra_overrides_extra=['++model.compile_image_encoder=false'])
    weights = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)['model']
    predictor.load_state_dict(weights, strict=True)
    del weights
    predictor.eval().cuda()
    predictor.fill_hole_area = 0
    install_rope_repeat_cache()
    team = TeamModel.load(base/'outputs/team_correction_mcbyte/.Han-Ber4_team.pkl', device='cuda')
    cache = load_detection_cache(base/'outputs/team_dataset/.Han-Ber4_detections_v1.npz')
    files = sorted((base/'data/cache/frames/Han-Ber4_cached').glob('*.jpg'), key=lambda x:int(x.stem))
    assert len(files) >= max(args.frames, args.warmup)
    video = base/'data/raw/Han-Ber4_cached.mp4'

    def drive(directory):
        return drive_sam2(video, lambda fid:person_detections(cache, fid), team,
            str(checkpoint), frame_cache_dir=directory, max_frames=args.frames,
            predictor_factory=lambda _:predictor, desc='SAM2 diagnostic')

    @contextmanager
    def bounded_cache(count):
        with tempfile.TemporaryDirectory(prefix='sam2-gpu-profile-') as tmp:
            for f in files[:count]:
                (Path(tmp)/f.name).symlink_to(f.resolve())
            yield Path(tmp)

    with bounded_cache(args.warmup) as directory:
        manager, seeds, frames = drive(directory)
        for result in frames:
            pass
        del manager, seeds, frames, result
    gc.collect()
    torch.cuda.synchronize()
    metadata = {'revision':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'driver_sha256':hashlib.sha256((ROOT/'src/handball_cv/tracking/sam2_driver.py').read_bytes()).hexdigest(),
        'upstream_revision':subprocess.check_output(['git','-C',str(base/'sam2-upstream'),'rev-parse','HEAD'],text=True).strip(),
        'device':torch.cuda.get_device_name(), 'torch':torch.__version__, 'cuda':torch.version.cuda,
        'frames':args.frames,'warmup_frames':args.warmup,'sample_frame':args.sample_frame,
        'precision':'BF16 autocast, eager; shared production driver and historical team appearance',
        'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True),
        'hardware_counters':'unavailable for current user: ERR_NVGPUCTRPERM; RmProfilingAdminOnly=1'}

    # Plain run first: no module hooks, SDPA wrappers, or profiler instrumentation.
    baseline_rows = []
    with bounded_cache(args.frames) as directory:
        manager, seeds, frames = drive(directory)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for result in frames:
            baseline_rows.append((result.frame_idx,result.player_ids.copy(),result.boxes.copy()))
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
    metadata['baseline']={'loop_seconds':elapsed,'loop_fps':len(baseline_rows)/elapsed,
        'returned_frames':len(baseline_rows),'mean_objects':float(np.mean([len(r[1]) for r in baseline_rows]))}
    print('BASELINE',json.dumps(metadata['baseline']),flush=True)

    active = False
    stack = []
    captures = {}
    layer_capture = []
    shapes = Counter()
    components = defaultdict(list)

    def wrap(module, label):
        original = module.forward
        @functools.wraps(original)
        def forward(*a, **kw):
            if not active:
                return original(*a, **kw)
            stack.append(label)
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            if label=='memory_attention.layer0' and not layer_capture:
                layer_capture.append((a,kw))
            start.record()
            with torch.profiler.record_function('sam2::'+label):
                torch.cuda.nvtx.range_push('sam2::'+label)
                try:
                    out = original(*a, **kw)
                finally:
                    torch.cuda.nvtx.range_pop()
            end.record()
            components[label].append((start,end))
            stack.pop()
            return out
        module.forward = forward

    for label in ['image_encoder','memory_attention','memory_encoder','sam_mask_decoder','sam_prompt_encoder']:
        wrap(getattr(predictor,label),label)
    for i,layer in enumerate(predictor.memory_attention.layers):
        wrap(layer, f'memory_attention.layer{i}')
        wrap(layer.self_attn,f'memory_attention.layer{i}.self_attn')
        wrap(layer.cross_attn_image,f'memory_attention.layer{i}.cross_attn')
        wrap(layer.linear1,f'memory_attention.layer{i}.ffn1')
        wrap(layer.linear2,f'memory_attention.layer{i}.ffn2')

    sdpa = F.scaled_dot_product_attention
    def capture_sdpa(q,k,v,*a,**kw):
        if active:
            label = stack[-1] if stack else 'other'
            shape = (label,tuple(q.shape),tuple(k.shape),str(q.dtype))
            shapes[shape] += 1
            if label.startswith('memory_attention.layer0.') and label not in captures:
                captures[label]=(q.detach(),k.detach(),v.detach(),a,kw)
        return sdpa(q,k,v,*a,**kw)
    F.scaled_dot_product_attention = capture_sdpa

    checks = []
    profiler = None
    with bounded_cache(args.frames) as directory:
        manager,seeds,frames = drive(directory)
        for fid in range(1,args.frames):
            if fid == args.sample_frame:
                torch.cuda.synchronize()
                if args.mode=='trace':
                    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],record_shapes=True,with_stack=False)
                    profiler.__enter__()
                active = True
            result = next(frames)
            if fid == args.sample_frame:
                torch.cuda.synchronize()
                active = False
                if profiler:
                    profiler.__exit__(None,None,None)
            ref = baseline_rows[fid-1]
            checks.append(result.frame_idx==ref[0] and np.array_equal(result.player_ids,ref[1])
                and np.array_equal(result.boxes,ref[2]))
        try:
            next(frames)
        except StopIteration:
            pass
    F.scaled_dot_product_attention = sdpa
    metadata['profile_matches_baseline_boxes_and_ids']=all(checks)
    metadata['sample_components_ms']={key:{'calls':len(values),'sum':sum(s.elapsed_time(e) for s,e in values)}
        for key,values in components.items()}
    metadata['sdpa_shapes']=[{'module':label,'q':q,'k':k,'dtype':dtype,'calls':count}
        for (label,q,k,dtype),count in shapes.items()]

    if profiler:
        profiler.export_chrome_trace(str(args.output/'frame_trace.json'))
        (args.output/'operators.txt').write_text(profiler.key_averages(group_by_input_shape=True).table(
            sort_by='self_cuda_time_total',row_limit=45))
        trace=json.loads((args.output/'frame_trace.json').read_text())
        kernels=defaultdict(lambda:[0,0.0])
        for event in trace['traceEvents']:
            if event.get('cat')=='kernel':
                kernels[event['name']][0]+=1
                kernels[event['name']][1]+=event['dur']
        metadata['kernels']=[{'name':k,'calls':v[0],'total_ms':v[1]/1000}
            for k,v in sorted(kernels.items(),key=lambda x:-x[1][1])]
        metadata['kernel_sum_ms']=sum(v[1] for v in kernels.values())/1000

    def timing(fn,repeats=12):
        for _ in range(4): fn()
        torch.cuda.synchronize()
        times=[]
        for _ in range(5):
            s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(repeats): fn()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e)/repeats)
        return {'median_ms':float(np.median(times)),'samples_ms':times}

    micro=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for label,(q,k,v,a,kw) in captures.items():
            measured=timing(lambda:sdpa(q,k,v,*a,**kw))
            b,h,m,d=q.shape
            n=k.shape[-2]
            flops=4*b*h*m*n*d  # QK and AV only; excludes softmax and indexing
            bytes_min=sum(t.numel()*t.element_size() for t in [q,k,v])+q.numel()*q.element_size()
            micro.append({'module':label,'q':list(q.shape),'k':list(k.shape),'strides_q':list(q.stride()),
                'strides_k':list(k.stride()),'qk_av_flops':flops,'unique_tensor_bytes_lower_bound':bytes_min,
                'algorithmic_flops_per_unique_byte':flops/bytes_min,
                'effective_qk_av_tflops':flops/(measured['median_ms']*1e9),**measured})
        if args.mode=='counters':
            assert layer_capture
            a,kw=layer_capture[0]
            for _ in range(3): predictor.memory_attention.layers[0](*a,**kw)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push('sam2_representative_memory_layer')
            predictor.memory_attention.layers[0](*a,**kw)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
    metadata['attention_microbenchmarks']=micro
    (args.output/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('RESULT',json.dumps({k:v for k,v in metadata.items() if k!='kernels'},indent=2),flush=True)


if __name__=='__main__':
    main()
