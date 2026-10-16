"""Overlap a chunk's next GPU frame with the current frame's CPU mask work.

Only the known drive_sam2 consumer enables prefetch: arbitrary predictor callers
may correct state between yields and must retain the synchronous API. A producer
owns the predictor state for exactly one propagation chunk, then joins before
checkpoint decisions can mutate it. At most two extra frames can be in flight.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from queue import Empty, Full, Queue
from threading import Event, Thread
from types import MethodType

import torch

_ACTIVE = ContextVar('sam2_cpu_overlap_driver', default=False)


def _prefetch(source_factory, stream, device, autocast_enabled, autocast_dtype):
    queue = Queue(maxsize=1)
    stop = Event()
    caller_stream = torch.cuda.current_stream(device)
    stream.wait_stream(caller_stream)

    def send(item):
        while not stop.is_set():
            try:
                queue.put(item, timeout=0.05)
                return True
            except Full:
                pass
        return False

    def produce():
        source = None
        try:
            with torch.cuda.device(device), torch.cuda.stream(stream), torch.inference_mode(), \
                    torch.autocast('cuda', enabled=autocast_enabled, dtype=autocast_dtype):
                source = iter(source_factory())
                for fid, ids, masks in source:
                    if stop.is_set():
                        break
                    ready = torch.cuda.Event()
                    ready.record(stream)
                    # Upstream ids aliases state; copy before crossing the thread boundary.
                    if not send(('frame', (fid, list(ids), masks), ready)):
                        break
                send(('end', None, None))
        except BaseException as error:
            send(('error', error, None))
        finally:
            if source is not None and hasattr(source, 'close'):
                source.close()

    worker = Thread(target=produce, name='sam2-propagate', daemon=True)
    worker.start()
    try:
        while True:
            try:
                kind, value, ready = queue.get(timeout=0.05)
            except Empty:
                if not worker.is_alive():
                    raise RuntimeError('SAM2 producer exited without an end marker')
                continue
            if kind == 'end':
                return
            if kind == 'error':
                raise value
            current = torch.cuda.current_stream(device)
            current.wait_event(ready)
            value[2].record_stream(current)
            yield value
    finally:
        stop.set()
        worker.join()
        # A caller that closes early can mutate state immediately after close.
        # Order any outstanding producer writes before that caller's work.
        torch.cuda.current_stream(device).wait_stream(stream)


def install(predictor):
    """Enable only forward CUDA propagation reached through drive_sam2."""
    import handball_cv.tracking.sam2_driver as driver
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    if type(predictor) is not SAM2VideoPredictor or getattr(predictor, '_cpu_overlap', False):
        return
    previous_propagate = predictor.propagate_in_video
    stream = None

    @wraps(previous_propagate)
    def propagate(self, inference_state, *args, **kwargs):
        nonlocal stream
        device = torch.device(inference_state['device'])
        reverse = kwargs.get('reverse', args[2] if len(args) > 2 else False)
        if (not _ACTIVE.get() or device.type != 'cuda' or reverse or self.training
                or not torch.is_inference_mode_enabled()):
            yield from previous_propagate(inference_state, *args, **kwargs)
            return
        if stream is None:
            stream = torch.cuda.Stream(device=device)
        yield from _prefetch(
            lambda: previous_propagate(inference_state, *args, **kwargs), stream, device,
            torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))

    predictor.propagate_in_video = MethodType(propagate, predictor)
    predictor._cpu_overlap = True
    if getattr(driver.drive_sam2, '_cpu_overlap_scope', False):
        return
    previous_drive = driver.drive_sam2

    @wraps(previous_drive)
    def drive(*args, **kwargs):
        token = _ACTIVE.set(True)
        try:
            manager, seeds, source = previous_drive(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)

        def frames():
            try:
                while True:
                    token = _ACTIVE.set(True)
                    try:
                        result = next(source)
                    except StopIteration:
                        return
                    finally:
                        _ACTIVE.reset(token)
                    yield result
            finally:
                source.close()
        return manager, seeds, frames()

    drive._cpu_overlap_scope = True
    driver.drive_sam2 = drive
