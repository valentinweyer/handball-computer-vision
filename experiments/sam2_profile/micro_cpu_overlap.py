"""Verify cross-stream dependencies, retained outputs, exceptions and early close."""
import argparse
import json
from pathlib import Path
import time
import torch
from experiments.sam2_profile.patches.cpu_overlap import _prefetch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(7)
    device = torch.device('cuda')
    x = torch.randn(1024, 1024, device=device)
    w = torch.randn(1024, 1024, device=device)
    stream = torch.cuda.Stream()
    def source():
        for i in range(8):
            yield i, [1, 2], (x + i) @ w
    with torch.inference_mode():
        ref = [z[2].clone() for z in source()]
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        outputs = [z[2] for z in _prefetch(source, stream, device, False, torch.bfloat16)]
        end.record()
        end.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(ref, outputs))
        early = _prefetch(source, stream, device, False, torch.bfloat16)
        next(early)
        early.close()
        torch.cuda.synchronize()
        def bad_source():
            yield 0, [1], x.clone()
            raise ValueError('producer failure')
        try:
            list(_prefetch(bad_source, stream, device, False, torch.bfloat16))
        except ValueError as error:
            assert str(error) == 'producer failure'
        else:
            raise AssertionError('producer exception was swallowed')
    out = {'identical': True, 'retained_outputs': True, 'early_close': True,
           'exception_forwarded': True, 'event_ms': start.elapsed_time(end)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out))

if __name__ == '__main__':
    main()
