"""Bounded GPU comparison using the unchanged production lifecycle manager."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter


def json_default(value):
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['felix', 'han', 'bhc'], default='felix')
    parser.add_argument('--backend', choices=['cutie', 'sam2', 'lean_sam2'], default='cutie')
    parser.add_argument('--lean-method', choices=['memory', 'off'], default='memory',
                        help='lean_sam2 only; see experiments/lean_sam2/adapter.py')
    parser.add_argument('--cutie-size', type=int, choices=[480, 1024], default=1024)
    parser.add_argument('--max-frames', type=int, default=141)
    parser.add_argument('--warmup-frames', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_frames < 2 or args.warmup_frames < 2:
        parser.error('Use at least two frames for measurement and warmup')
    if args.output.exists():
        parser.error('Output directory must be new')
    from experiments.cutie.adapter import (
        ROOT, BASE, UPSTREAM, SAM2_UPSTREAM, CHECKPOINT, SAM2_CHECKPOINT,
        bootstrap, sha256, build_models,
    )
    bootstrap()
    if args.backend == 'lean_sam2':
        from experiments.lean_sam2 import adapter as lean
        lean.bootstrap()
    from experiments.cutie.cases import inputs
    import numpy as np
    import torch
    from handball_cv.teams.model import TeamModel
    from handball_cv.tracking.sam2_driver import drive_sam2, init_state_lazily, install_rope_repeat_cache
    from scripts.render_raw_team_classification import load_detection_cache, person_detections
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    torch.manual_seed(0)
    paths = inputs(args.case)
    frame_dir = paths['frames']
    all_files = sorted(frame_dir.glob('*.jpg'), key=lambda p: int(p.stem))
    if len(all_files) < max(args.max_frames, args.warmup_frames):
        parser.error('Requested prefix exceeds available frames')
    assert [int(path.stem) for path in all_files[:args.max_frames]] == list(range(args.max_frames))
    video, detections, team_file = paths['video'], paths['detections'], paths['team']
    checkpoint = CHECKPOINT if args.backend == 'cutie' else SAM2_CHECKPOINT
    manifest = {'backend': args.backend, 'case': args.case, 'status': 'initializing',
                'processed_frames': 0, 'requested_frames': args.max_frames,
                'protocol': 'shared production TrackManager; periodic RF-DETR corrections',
                'video_sha256': sha256(video), 'detections_sha256': sha256(detections),
                'team_model_sha256': sha256(team_file), 'checkpoint_sha256': sha256(checkpoint),
                'sam2_image_checkpoint_sha256': sha256(SAM2_CHECKPOINT),
                'driver_sha256': sha256(ROOT/'src/handball_cv/tracking/sam2_driver.py'),
                'adapter_sha256': sha256(Path(__file__).with_name('adapter.py')),
                'runner_sha256': sha256(__file__),
                'cases_sha256': sha256(Path(__file__).with_name('cases.py')), 'project_revision': subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'upstream_patch_sha256': sha256(Path(__file__).with_name('upstream-delete.patch')),
                'cutie_revision': subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
                'sam2_revision': subprocess.check_output(['git', '-C', str(SAM2_UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
                'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'precision': 'BF16 autocast', 'check_every': 10, 'checkpoint_policy': 'reprompt',
                'warmup_frames': args.warmup_frames, 'live_detection': False, 'ocr': False,
                'reid': 'historical team-model features, matching the SAM2 tracking-only baseline',
                'capture_masks': False, 'scope': 'tracking only; model build, warmup, artifact writing excluded; seed and loop reported separately and together'}
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    started = perf_counter()
    if args.backend == 'cutie':
        predictor, loading = build_models(max_internal_size=args.cutie_size)
        manifest['strict_loading'] = loading
        manifest['cutie_config'] = __import__('omegaconf').OmegaConf.to_container(predictor.model.cfg, resolve=True)
        manifest['prompt_conversion'] = 'SAM2.1-L image-only box head; Cutie propagates every frame'
    elif args.backend == 'lean_sam2':
        # The fork's own builder; drive_sam2 serves it LazyFrames like production.
        predictor, values = lean.build_predictor(SAM2_CHECKPOINT, args.lean_method)
        manifest['lean_sam2'] = {'revision': lean.revision(), 'method': args.lean_method,
                                 'model_arguments': values, 'tarr_routing': False}
        manifest.pop('sam2_revision')
    else:
        from sam2.build_sam import build_sam2_video_predictor
        native = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',
                    ckpt_path=None, device='cpu', vos_optimized=False,
                    hydra_overrides_extra=['++model.compile_image_encoder=false'])
        weights = torch.load(SAM2_CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)['model']
        native.load_state_dict(weights, strict=True)
        manifest['strict_loading'] = {'sam2': {'tensors': len(weights), 'missing_keys': [], 'unexpected_keys': []}}
        del weights
        native.eval().cuda()
        native.fill_hole_area = 0
        install_rope_repeat_cache()
        class LazyPredictor:
            def init_state(self, video_path):
                return init_state_lazily(native, Path(video_path))
            def __getattr__(self, name):
                return getattr(native, name)
        predictor = LazyPredictor()
    torch.cuda.synchronize()
    manifest['model_build_seconds'] = perf_counter()-started
    team = TeamModel.load(team_file, device='cuda')
    cache = load_detection_cache(detections)
    assert set(np.unique(cache['class_id'])) <= {1, 2}
    def drive(directory):
        return drive_sam2(video, lambda fid: person_detections(cache, fid), team,
                          str(checkpoint), frame_cache_dir=directory,
                          predictor_factory=lambda _: predictor, desc=f'{args.backend} {args.case}')
    def link_frames(directory, count):
        for path in all_files[:count]:
            (Path(directory)/path.name).symlink_to(path.resolve())
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix='cutie-warmup-') as tmp:
        link_frames(tmp, args.warmup_frames)
        manager, seeds, frames = drive(Path(tmp))
        for result in frames:
            pass
        del manager, seeds, frames, result
    if args.backend == 'cutie':
        del predictor.last_state
    gc.collect()
    torch.cuda.synchronize()
    manifest['warmup_seconds'] = perf_counter()-started
    torch.cuda.reset_peak_memory_stats()
    rows, stats, processed = [], [], 0
    loop_started = None
    try:
        with tempfile.TemporaryDirectory(prefix='cutie-clip-') as tmp:
            link_frames(tmp, args.max_frames)
            total_started = perf_counter()
            manager, seeds, frames = drive(Path(tmp))
            rows.extend((0, int(oid), box.copy()) for oid, box in seeds.items())
            processed = 1
            torch.cuda.synchronize()
            manifest['fresh_state_seed_seconds'] = perf_counter()-total_started
            loop_started = previous = perf_counter()
            for result in frames:
                now = perf_counter()
                stats.append({'frame': result.frame_idx, 'objects': len(result.player_ids),
                              'resume_seconds': now-previous})
                rows.extend((result.frame_idx, int(oid), box.copy())
                            for oid, box in zip(result.player_ids, result.boxes))
                processed += 1
                previous = perf_counter()
            torch.cuda.synchronize()
            manifest['loop_seconds'] = perf_counter()-loop_started
            manifest['total_tracking_seconds'] = perf_counter()-total_started
            manifest['loop_fps'] = len(stats)/manifest['loop_seconds']
            manifest['total_tracking_fps'] = processed/manifest['total_tracking_seconds']
            manifest['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            manifest['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            assert processed == args.max_frames
            manifest['status'] = 'complete'
            if args.backend == 'cutie':
                manifest['adapter_steps'] = predictor.stats
            (args.output/'events.json').write_text(json.dumps(manager.events, indent=2, default=json_default)+'\n')
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest['processed_frames'] = processed
        np.savez_compressed(args.output/'replay.npz',
                            frame_index=np.array([row[0] for row in rows], dtype=int),
                            tracker_id=np.array([row[1] for row in rows], dtype=int),
                            boxes=np.array([row[2] for row in rows], dtype=float).reshape(-1, 4))
        (args.output/'frames.json').write_text(json.dumps(stats, indent=2)+'\n')
        manifest_path.write_text(json.dumps(manifest, indent=2, default=json_default)+'\n')
    print(json.dumps({key: manifest[key] for key in ['backend', 'status', 'processed_frames',
                  'loop_seconds', 'loop_fps', 'total_tracking_fps']}, indent=2))


if __name__ == '__main__':
    main()
