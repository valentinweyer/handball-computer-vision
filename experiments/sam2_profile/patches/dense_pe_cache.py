"""Bounded exact cache for the inference-only SAM prompt dense position map.

Dense position encoding depends on the Gaussian buffer, image grid and math
context, never on frame/object state. Keep one result; training, gradients,
inference tensors without a mutation version, and unknown encoders fall back.
"""
from __future__ import annotations

from types import MethodType

import torch


def _install_encoder(encoder):
    if getattr(encoder, '_dense_pe_cache_installed', False):
        return
    previous = encoder.get_dense_pe
    entry = None

    def get_dense_pe(self):
        nonlocal entry
        if self.training or torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
            return previous()
        buffer = self.pe_layer.positional_encoding_gaussian_matrix
        try:
            version = buffer._version
        except RuntimeError:  # tensors constructed under inference_mode have no version
            return previous()
        device_type = buffer.device.type
        if device_type not in ('cpu', 'cuda'):
            return previous()
        autocast = torch.is_autocast_enabled(device_type)
        autocast_dtype = torch.get_autocast_dtype(device_type) if autocast else None
        key = (id(buffer), buffer.data_ptr(), version, buffer.device, buffer.dtype,
               tuple(self.image_embedding_size), autocast, autocast_dtype,
               torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
        if entry is None or entry[0] != key:
            result = previous()
            # Retain the source buffer to prevent id/data_ptr reuse after replacement.
            event, producer_stream = None, None
            if result.is_cuda:
                stream = torch.cuda.current_stream(result.device)
                producer_stream = stream.cuda_stream
                event = torch.cuda.Event()
                event.record(stream)
            entry = (key, result, buffer, event, producer_stream)
        elif entry[3] is not None:
            # Same-stream use is ordered already; side-stream users need readiness
            # and allocator lifetime tracking if this entry is later replaced.
            stream = torch.cuda.current_stream(entry[1].device)
            if stream.cuda_stream != entry[4]:
                stream.wait_event(entry[3])
                entry[1].record_stream(stream)
        # Preserve the public API: callers may mutate a returned position map.
        # The private cached tensor must never escape.
        return entry[1].clone()

    encoder.get_dense_pe = MethodType(get_dense_pe, encoder)
    encoder._dense_pe_cache_installed = True


def install(predictor):
    """ab.py entry point; preserve arbitrary third-party predictor behavior."""
    from sam2.modeling.sam.prompt_encoder import PromptEncoder
    encoder = getattr(predictor, 'sam_prompt_encoder', None)
    if type(encoder) is PromptEncoder:
        _install_encoder(encoder)
