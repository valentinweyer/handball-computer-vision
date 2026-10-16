"""CPU regression proof for small-graph/overlap source coordination.

Run as ``python -m experiments.sam2_profile.micro_graph_coordination``.
The real coordinator wraps a synchronous generator transport in place of the
CUDA prefetch transport. No CUDA APIs or model inference execute. This checks
buffering and generator control flow; GPU synchronization remains a stack-test
responsibility.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import weakref

from experiments.sam2_profile.patches import cpu_overlap
from experiments.sam2_profile.patches.cuda_small_graphs import (
    _CAPTURE_ALLOWED,
    _SERIAL_PRODUCER,
    _coordinate_cpu_overlap,
)


def _cpu_prefetch(factory, *args, **kwargs):
    source = iter(factory())
    try:
        yield from source
    finally:
        source.close()


def _assert_default_context():
    assert _CAPTURE_ALLOWED.get() is True
    assert _SERIAL_PRODUCER.get() is False


def run_checks():
    original_prefetch = cpu_overlap._prefetch
    try:
        cpu_overlap._prefetch = _cpu_prefetch
        _coordinate_cpu_overlap()
        wrapped = cpu_overlap._prefetch
        _coordinate_cpu_overlap()
        assert cpu_overlap._prefetch is wrapped, 'Coordinator was wrapped twice'

        for count in (0, 1, 2, 4):
            trace, closed = [], []

            def source():
                try:
                    for index in range(count):
                        trace.append((index, _CAPTURE_ALLOWED.get(), _SERIAL_PRODUCER.get()))
                        yield index
                finally:
                    closed.append(True)

            frames = wrapped(source, None)
            if count:
                assert next(frames) == 0
                assert len(trace) == min(count, 2), 'First result escaped before prebuffering'
                _assert_default_context()
                assert [0] + list(frames) == list(range(count))
            else:
                assert list(frames) == []
            assert trace == [(i, i < 2, True) for i in range(count)]
            assert closed == [True], 'Source close count changed'
            _assert_default_context()

        # Closing while the first result is exposed closes the source and drops
        # its second buffered result without advancing to a third one.
        trace, closed = [], []
        live = weakref.WeakSet()

        class Frame:
            def __init__(self, index):
                self.index = index

        def early_source():
            try:
                for index in range(5):
                    trace.append(index)
                    frame = Frame(index)
                    live.add(frame)
                    yield frame
            finally:
                closed.append(True)

        frames = wrapped(early_source, None)
        first = next(frames)
        assert first.index == 0 and trace == [0, 1]
        assert len(live) == 2
        frames.close()
        gc.collect()
        assert closed == [True] and trace == [0, 1]
        assert list(live) == [first], 'Buffered frame survived close'
        del first
        gc.collect()
        assert len(live) == 0
        _assert_default_context()

        # An error while buffering result two must follow result one at the
        # public iterator boundary. Later errors obey the same ordering.
        for fail_after in (0, 1, 2, 3):
            closed = []

            def bad_source():
                try:
                    for index in range(fail_after):
                        yield index
                    raise ValueError('expected producer failure')
                finally:
                    closed.append(True)

            frames = wrapped(bad_source, None)
            received = []
            try:
                while True:
                    received.append(next(frames))
            except ValueError as error:
                assert str(error) == 'expected producer failure'
            else:
                raise AssertionError('Producer error was swallowed')
            assert received == list(range(fail_after)), 'Buffered result/error ordering changed'
            assert closed == [True]
            _assert_default_context()

        # Restore the caller's actual ContextVar values, including non-default
        # scopes; the coordinator must never reset directly to a guessed default.
        capture_token = _CAPTURE_ALLOWED.set(False)
        serial_token = _SERIAL_PRODUCER.set(True)
        try:
            observed = []

            def scoped_source():
                for index in range(4):
                    observed.append((_CAPTURE_ALLOWED.get(), _SERIAL_PRODUCER.get()))
                    yield index

            frames = wrapped(scoped_source, None)
            for index in range(4):
                assert next(frames) == index
                assert _CAPTURE_ALLOWED.get() is False and _SERIAL_PRODUCER.get() is True
            frames.close()
            assert observed == [(True, True), (True, True), (False, True), (False, True)]
        finally:
            _SERIAL_PRODUCER.reset(serial_token)
            _CAPTURE_ALLOWED.reset(capture_token)
        _assert_default_context()

        # A new chunk gets a fresh capture window even after prior exhaustion,
        # failures and early closes.
        restart_trace = []

        def restart_source():
            for index in range(4):
                restart_trace.append(_CAPTURE_ALLOWED.get())
                yield index

        assert list(wrapped(restart_source, None)) == [0, 1, 2, 3]
        assert restart_trace == [True, True, False, False]
        _assert_default_context()
        return {'cpu_only': True, 'source_lengths': [0, 1, 2, 4],
                'two_advances_before_first_exposure': True, 'idempotent': True,
                'early_close': True, 'buffered_references_released': True,
                'ordered_errors_after_results': [0, 1, 2, 3],
                'context_restored': True, 'nondefault_context_restored': True,
                'chunk_capture_window_reset': True}
    finally:
        cpu_overlap._prefetch = original_prefetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run_checks()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
