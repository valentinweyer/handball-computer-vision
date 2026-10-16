"""Skip driver-discarded prompt previews; preserve all SAM2 state updates.

The production driver ignores every add_new_* / remove_object return value.
Upstream has already stored the prompt output before building its full-video-
resolution editing preview. Suppress that preview only while the known driver
is executing; direct predictor callers keep the complete upstream API.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from types import MethodType

_DRIVER_ACTIVE = ContextVar('sam2_profile_driver_ignores_prompt_returns', default=False)
_PROMPT_PREVIEW_UNUSED = ContextVar('sam2_profile_prompt_preview_unused', default=False)
_NO_PREVIEW = object()


def _wrap_driver(driver):
    if getattr(driver.drive_sam2, '_checkpoint_outputs_context', False):
        return
    original = driver.drive_sam2

    @wraps(original)
    def drive(*args, **kwargs):
        token = _DRIVER_ACTIVE.set(True)
        try:
            manager, seeds, frames = original(*args, **kwargs)
        finally:
            _DRIVER_ACTIVE.reset(token)

        def iterate():
            try:
                while True:
                    token = _DRIVER_ACTIVE.set(True)
                    try:
                        item = next(frames)
                    except StopIteration:
                        return
                    finally:
                        _DRIVER_ACTIVE.reset(token)
                    yield item
            finally:
                close = getattr(frames, 'close', None)
                if close is not None:
                    close()

        return manager, seeds, iterate()

    drive._checkpoint_outputs_context = True
    driver.drive_sam2 = drive


def _patch_predictor(predictor):
    if getattr(predictor, '_checkpoint_outputs_installed', False):
        return
    consolidate = predictor._consolidate_temp_output_across_obj
    resize = predictor._get_orig_video_res_output
    remove = predictor.remove_object

    def consolidate_without_preview(self, *args, **kwargs):
        if _PROMPT_PREVIEW_UNUSED.get() and kwargs.get('consolidate_at_video_res') is True:
            return {'pred_masks_video_res': _NO_PREVIEW}
        return consolidate(*args, **kwargs)

    def resize_without_preview(self, state, masks):
        if _PROMPT_PREVIEW_UNUSED.get() and masks is _NO_PREVIEW:
            return None, None
        return resize(state, masks)

    def remove_without_preview(self, state, obj_id, strict=False, need_output=True):
        return remove(state, obj_id, strict=strict,
                      need_output=False if _DRIVER_ACTIVE.get() else need_output)

    predictor._consolidate_temp_output_across_obj = MethodType(consolidate_without_preview, predictor)
    predictor._get_orig_video_res_output = MethodType(resize_without_preview, predictor)
    predictor.remove_object = MethodType(remove_without_preview, predictor)

    for name in ('add_new_points_or_box', 'add_new_mask'):
        previous = getattr(predictor, name)

        def wrap_prompt(original):
            @wraps(original)
            def prompt(self, *args, **kwargs):
                token = _PROMPT_PREVIEW_UNUSED.set(_DRIVER_ACTIVE.get())
                try:
                    return original(*args, **kwargs)
                finally:
                    _PROMPT_PREVIEW_UNUSED.reset(token)
            return prompt

        setattr(predictor, name, MethodType(wrap_prompt(previous), predictor))
    predictor._checkpoint_outputs_installed = True


def install(predictor):
    """ab.py entry point; unknown predictor implementations stay unchanged."""
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    if type(predictor) is not SAM2VideoPredictor:
        return
    import handball_cv.tracking.sam2_driver as driver
    _wrap_driver(driver)
    _patch_predictor(predictor)
