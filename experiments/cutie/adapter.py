"""Cutie propagation with SAM2 image-only box prompts and the shared manager."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/valentinweyer/projects/handball-computer-vision')
UPSTREAM = ROOT / '.local/cutie/upstream'
SAM2_UPSTREAM = BASE / 'sam2-upstream'
CHECKPOINT = BASE / 'models/cutie/cutie-base-mega.pth'
SAM2_CHECKPOINT = BASE / 'segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt'


def bootstrap():
    # The repository's scripts package must precede Cutie's training scripts.
    for path in [SAM2_UPSTREAM, UPSTREAM, ROOT / 'src', ROOT]:
        sys.path.insert(0, str(path))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build_models(device='cuda', max_internal_size=1024):
    bootstrap()
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from cutie.model.cutie import CUTIE
    if device not in {'cpu', 'cuda'}:
        raise ValueError('Use explicit cpu or cuda')
    if device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    # Generic-video memory settings, retaining the native architecture and weights.
    with initialize_config_dir(version_base='1.3', config_dir=str(UPSTREAM / 'cutie/config')):
        cfg = compose(config_name='eval_config', overrides=[
            'use_long_term=true', 'mem_every=5', f'max_internal_size={max_internal_size}'])
    cutie = CUTIE(cfg)
    weights = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    cutie.load_state_dict(weights, strict=True)
    loaded = {'cutie': {'tensors': len(weights), 'missing_keys': [], 'unexpected_keys': []}}
    del weights
    cutie.eval().to(device)
    # Cutie's config context precedes SAM2's package-owned Hydra initialization.
    GlobalHydra.instance().clear()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', ckpt_path=None,
                    device='cpu', apply_postprocessing=True)
    weights = torch.load(SAM2_CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)['model']
    sam.load_state_dict(weights, strict=True)
    loaded['sam2_image_head'] = {'tensors': len(weights), 'missing_keys': [], 'unexpected_keys': []}
    del weights
    sam.eval().to(device)
    assert all(p.device.type == device for model in [sam, cutie] for p in model.parameters())
    return CutieAdapter(cutie, SAM2ImagePredictor(sam)), loaded


class CutieAdapter:
    """Minimal SAM2-driver interface; no SAM2 temporal propagation is invoked.

    Prompts at the same boundary are batched into one Cutie mask observation.
    An unchanged repeated boundary returns its cached result. A corrected boundary
    gets one additional native Cutie step on that same image, retaining memory.
    """
    def __init__(self, model, seeder):
        self.model, self.seeder = model, seeder
        self.device = next(model.parameters()).device
        self.stats = {}

    def init_state(self, video_path):
        from cutie.inference.inference_core import InferenceCore
        state = {'files': sorted(Path(video_path).glob('*.jpg'), key=lambda p: int(p.stem)),
                 'core': InferenceCore(self.model, cfg=self.model.cfg),
                 'pending': {}, 'obj_id_to_idx': {}, 'last_frame': -1,
                 'cached': None, 'rgb_frame': -1, 'rgb': None}
        self.stats = {'cutie_steps': 0, 'boundary_correction_steps': 0,
                      'box_prompt_batches': 0, 'box_prompt_objects': 0}
        self.last_state = state
        return state

    def _refresh_ids(self, state):
        ids = state['core'].object_manager.all_obj_ids
        ids = ids + [oid for oid in state['pending'] if oid not in ids]
        state['obj_id_to_idx'] = {oid: index for index, oid in enumerate(ids)}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box, **kwargs):
        state['pending'][int(obj_id)] = (int(frame_idx), 'box', box)
        self._refresh_ids(state)

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        state['pending'][int(obj_id)] = (int(frame_idx), 'mask', mask)
        self._refresh_ids(state)

    def remove_object(self, state, obj_id):
        core = state['core']
        obj_id = int(obj_id)
        state['pending'].pop(obj_id, None)
        before = core.object_manager.all_obj_ids
        if obj_id in before:
            keep = [index for index, oid in enumerate(before) if oid != obj_id]
            core.delete_objects([obj_id])
            # Native deletion renumbers channels but leaves this tensor unchanged.
            # Keep the previous-mask channels aligned with the surviving IDs.
            if core.last_mask is not None:
                core.last_mask = core.last_mask[:, keep]
            # Native purge does not delete object summaries; resets must start fresh.
            core.memory.obj_v.pop(obj_id, None)
        self._refresh_ids(state)
        if state['cached'] is not None:
            fid, ids, logits = state['cached']
            keep = [index for index, oid in enumerate(ids) if oid != obj_id]
            state['cached'] = (fid, [ids[index] for index in keep], logits[keep])

    def reset_state(self, state):
        from cutie.inference.inference_core import InferenceCore
        state['core'] = InferenceCore(self.model, cfg=self.model.cfg)
        state.update(pending={}, obj_id_to_idx={}, last_frame=-1, cached=None,
                     rgb_frame=-1, rgb=None)

    def _image(self, state, fid):
        import cv2
        if state['rgb_frame'] != fid:
            bgr = cv2.imread(str(state['files'][fid]))
            if bgr is None:
                raise ValueError(f'Unreadable frame {fid}')
            state['rgb'] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            state['rgb_frame'] = fid
        return state['rgb']

    def _prompt(self, state, fid, rgb):
        import numpy as np
        import torch
        pending = state['pending']
        assert all(item[0] == fid for item in pending.values()), 'Prompt frame mismatch'
        # Native Cutie requires updates ordered by existing channel, then new IDs.
        current = state['core'].object_manager.all_obj_ids
        order = [oid for oid in current if oid in pending]
        order += [oid for oid in pending if oid not in current]
        box_ids = [oid for oid in order if pending[oid][1] == 'box']
        masks = {}
        if box_ids:
            self.seeder.set_image(rgb)
            _, coords, labels, boxes = self.seeder._prep_prompts(
                None, None, np.stack([pending[oid][2] for oid in box_ids]), None, True)
            logits, _, _ = self.seeder._predict(coords, labels, boxes, None,
                                              multimask_output=False, return_logits=True)
            logits = logits.reshape(len(box_ids), *rgb.shape[:2])
            masks.update(zip(box_ids, logits))
            self.stats['box_prompt_batches'] += 1
            self.stats['box_prompt_objects'] += len(box_ids)
        for oid in order:
            if pending[oid][1] == 'mask':
                mask = torch.as_tensor(pending[oid][2], device=self.device, dtype=torch.bool)
                masks[oid] = mask.float()*2-1
        # Highest positive SAM logit wins overlapping seed pixels; background is 0.
        stack = torch.stack([masks[oid].float() for oid in order])
        value, index = stack.max(dim=0)
        ids = torch.tensor(order, device=self.device, dtype=torch.long)
        merged = torch.where(value > 0, ids[index], 0)
        state['pending'] = {}
        return merged, order

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track):
        import numpy as np
        import torch
        end = min(len(state['files'])-1, start_frame_idx+max_frame_num_to_track)
        for fid in range(start_frame_idx, end+1):
            if fid == state['last_frame'] and not state['pending']:
                yield state['cached']
                continue
            assert fid in {state['last_frame'], state['last_frame']+1}, 'Only causal frame order is supported'
            rgb = self._image(state, fid)
            image = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1).float()/255
            mask, objects = self._prompt(state, fid, rgb) if state['pending'] else (None, None)
            core = state['core']
            if core.object_manager.num_obj == 0 and mask is None:
                ids = []
                logits = torch.empty((0, 1, *rgb.shape[:2]), device=self.device, dtype=torch.int8)
            else:
                probability = core.step(image, mask, objects=objects)
                assert torch.isfinite(probability).all()
                indexed = probability.argmax(dim=0)
                ids = core.object_manager.all_obj_ids
                channels = torch.arange(1, len(ids)+1, device=self.device)[:, None, None]
                logits = ((indexed[None] == channels).to(torch.int8)*2-1)[:, None]
                self.stats['cutie_steps'] += 1
                self.stats['boundary_correction_steps'] += int(fid == state['last_frame'])
            state['last_frame'] = fid
            state['cached'] = (fid, ids, logits)
            self._refresh_ids(state)
            yield state['cached']
