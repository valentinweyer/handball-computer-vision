"""Validate loading and synthetic inputs only; never opens a real clip."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from experiments.cameltrack.runtime import CamelTracker, POSE_CHECKPOINT, provenance
    import numpy as np
    import onnxruntime as ort
    import torch

    tracker = CamelTracker(args.device)
    # Profile the real pose graph to distinguish CUDA kernels from a CPU fallback.
    options = ort.SessionOptions()
    options.enable_profiling = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options.profile_file_prefix = str(args.output.parent / 'pose-profile')
    provider = 'CUDAExecutionProvider' if args.device == 'cuda' else 'CPUExecutionProvider'
    tracker.pose.session = ort.InferenceSession(str(POSE_CHECKPOINT), options, providers=[provider])
    assert tracker.pose.session.get_providers()[0] == provider
    tracker.pose.session.disable_fallback()
    frame = np.full((384, 640, 3), 127, np.uint8)
    boxes = np.array([[80, 30, 190, 360], [350, 30, 470, 360]], np.float32)
    confidence = np.array([.95, .95], np.float32)
    steps = []
    for index in range(3):
        ids, output = tracker.update(frame, boxes, confidence)
        assert len(ids) == 2 and len(set(ids)) == 2
        assert np.isfinite(output).all()
        steps.append({'step': index, 'ids': ids.tolist()})
    ids, output = tracker.update(frame, [], [])
    assert len(ids) == 0 and output.shape == (0, 4)
    ids, _ = tracker.update(frame, boxes, confidence)
    assert len(ids) == 2
    tracker.reset()
    ids, _ = tracker.update(frame, boxes, confidence)
    assert ids.tolist() == steps[0]['ids']
    events = json.loads(Path(tracker.pose.session.end_profiling()).read_text())
    nodes = [event for event in events if event.get('args', {}).get('provider')]
    providers = Counter(event['args']['provider'] for event in nodes)
    conv_nodes = [event for event in nodes if 'conv' in event['args'].get('op_name', '').lower()]
    assert conv_nodes and all(event['args']['provider'] == provider for event in conv_nodes)
    if args.device == 'cuda':
        assert providers['CUDAExecutionProvider'] > 0
    else:
        assert not torch.cuda.is_initialized()
    report = {'status': 'passed', 'input': 'synthetic constant image; two boxes; no clip',
              'device': args.device, 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
              'onnxruntime': ort.__version__, 'loading': tracker.loading,
              'pose_node_executions_by_provider': dict(providers),
              'pose_convolutions_on_requested_device': True, 'steps': steps,
              'empty_frame': 'passed', 'reappearance': 'passed', 'reset': 'passed',
              **provenance()}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
