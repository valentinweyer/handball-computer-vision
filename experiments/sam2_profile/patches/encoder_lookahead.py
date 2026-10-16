"""Prefetch one forward image encoder result on a private CUDA stream.

Only official eager SAM2 inference under CUDA BF16 autocast is eligible.
The temporal/object state is untouched: the current frame stays in SAM2's
cache and one independent future image/backbone tuple waits alongside it.
Reset, a new state, reverse/out-of-order access, or a math-context change
invalidates the prefetch and follows the original implementation.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from types import MethodType

import torch

_TOKEN = '_sam2_profile_encoder_lookahead_session'
_FORWARD_ALLOWED = ContextVar('sam2_profile_encoder_forward_direction', default=True)


def _record_stream(value, stream):
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record_stream(item, stream)


def _install_predictor(predictor):
    if getattr(predictor, '_encoder_lookahead_installed', False):
        return
    original = predictor._get_image_feature
    reset = predictor.reset_state
    propagate = predictor.propagate_in_video
    pending = None
    streams = {}  # normally exactly one device; no session/tensor references
    counters = {'scheduled': 0, 'consumed': 0, 'invalidated': 0}

    def invalidate():
        nonlocal pending
        if pending is not None:
            # Order a future fallback encoder call after the outstanding encoder.
            stream = torch.cuda.current_stream(pending['device'])
            stream.wait_event(pending['event'])
            pending = None
            counters['invalidated'] += 1

    def get_image_feature(self, state, frame_idx, batch_size):
        nonlocal pending
        device = torch.device(state['device'])
        eligible = (_FORWARD_ALLOWED.get() and device.type == 'cuda' and not self.training
                    and torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
                    and torch.is_autocast_enabled('cuda')
                    and torch.get_autocast_dtype('cuda') == torch.bfloat16)
        if not eligible:
            invalidate()
            return original(state, frame_idx, batch_size)
        token = state.setdefault(_TOKEN, object())
        context = (token, id(state['images']), device, state['num_frames'],
                   torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
        if pending is not None:
            if pending['context'] != context or frame_idx > pending['target'] or frame_idx < pending['anchor']:
                invalidate()
            elif frame_idx == pending['target']:
                stream = torch.cuda.current_stream(device)
                stream.wait_event(pending['event'])
                _record_stream((pending['image'], pending['backbone']), stream)
                state['cached_features'] = {frame_idx: (pending['image'], pending['backbone'])}
                pending = None
                counters['consumed'] += 1

        result = original(state, frame_idx, batch_size)
        # A pending target belongs to the same frame during all per-object calls;
        # never compute it more than once or replace the current frame's cache.
        target = int(frame_idx) + 1
        if pending is None and target < state['num_frames']:
            caller = torch.cuda.current_stream(device)
            stream = streams.get(device)
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                streams[device] = stream
            stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                # LazyFrames is indexed on this same Python thread, so its small
                # decode/cache dictionaries are never accessed concurrently.
                image = state['images'][target].to(device).float().unsqueeze(0)
                backbone = self.forward_image(image)
                event = torch.cuda.Event()
                event.record(stream)
            pending = {'context': context, 'anchor': int(frame_idx), 'target': target,
                       'device': device, 'image': image, 'backbone': backbone, 'event': event}
            counters['scheduled'] += 1
        return result

    def reset_state(self, state):
        # Clear all lookahead work even for an old state: the predictor retains at
        # most one future image tuple across any number of sessions.
        invalidate()
        state.pop(_TOKEN, None)
        return reset(state)

    @wraps(propagate)
    def propagate_in_video(self, inference_state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False):
        frames = propagate(inference_state, start_frame_idx=start_frame_idx,
                           max_frame_num_to_track=max_frame_num_to_track, reverse=reverse)
        try:
            while True:
                token = _FORWARD_ALLOWED.set(not reverse)
                try:
                    item = next(frames)
                except StopIteration:
                    return
                finally:
                    _FORWARD_ALLOWED.reset(token)
                yield item
        finally:
            frames.close()

    predictor._get_image_feature = MethodType(get_image_feature, predictor)
    predictor.propagate_in_video = MethodType(propagate_in_video, predictor)
    predictor.reset_state = MethodType(reset_state, predictor)
    predictor._encoder_lookahead_installed = True
    predictor._encoder_lookahead_counters = counters


def install(predictor):
    """ab.py entry point; preserve unknown predictors and compiled subclasses."""
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    if type(predictor) is SAM2VideoPredictor:
        _install_predictor(predictor)
