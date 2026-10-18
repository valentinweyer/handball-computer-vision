"""Official CAMEL association/lifecycle, native KPR and RTMPose feature models."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / '.local/cameltrack'
CHECKPOINTS = ASSETS / 'checkpoints'
CAMEL_CHECKPOINT = CHECKPOINTS / 'camel_bbox_app_kps_sportsmot.ckpt'
KPR_CHECKPOINT = CHECKPOINTS / 'kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar'
POSE_CHECKPOINT = CHECKPOINTS / 'rtmpose-m.onnx'


def bootstrap():
    os.environ.setdefault('WANDB_MODE', 'disabled')
    for path in [ROOT / 'src', ASSETS / 'upstream', ASSETS / 'tracklab', ASSETS / 'kpr']:
        if not path.is_dir():
            raise FileNotFoundError(f'Missing source: {path}; see experiment README')
        sys.path.insert(0, str(path))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def provenance():
    import subprocess
    return {
        'source_revisions': {name: subprocess.check_output(
            ['git', '-C', str(ASSETS / name), 'rev-parse', 'HEAD'], text=True).strip()
            for name in ['upstream', 'tracklab', 'kpr']},
        'checkpoint_sha256': {p.name: sha256(p) for p in
                              [CAMEL_CHECKPOINT, KPR_CHECKPOINT, POSE_CHECKPOINT]},
        'runtime_sha256': sha256(__file__),
    }


class CamelTracker:
    """Use checkpoint-compatible appearance/pose cues; never replace them by zeros."""

    def __init__(self, device='cuda'):
        bootstrap()
        import torch
        from cameltrack import CAMELTrack
        from omegaconf import OmegaConf
        import torchreid
        from torchreid.data.transforms import build_transforms
        from torchreid.utils.tools import extract_test_embeddings
        from rtmlib import RTMPose

        if device not in {'cpu', 'cuda'}:
            raise ValueError('Use explicit cpu or cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        self.device = device
        self.extract_embeddings = extract_test_embeddings
        self.loading = {}
        for path in [CAMEL_CHECKPOINT, KPR_CHECKPOINT, POSE_CHECKPOINT]:
            if not path.is_file():
                raise FileNotFoundError(path)

        saved = torch.load(CAMEL_CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
        config = OmegaConf.create({'_target_': 'cameltrack.CAMEL', **saved['hyper_parameters']})
        # Load once on CPU, strictly, before transferring the complete model.
        # These are the paper's SportsMOT thresholds, not handball-tuned values.
        self.tracker = CAMELTrack(
            CAMEL=config, device='cpu', min_det_conf=0.1, min_init_det_conf=0.4,
            min_num_hits=0, max_wo_hits=150, max_track_gallery_size=50,
        )
        self.tracker.CAMEL.load_state_dict(saved['state_dict'], strict=True)
        self.loading['association'] = {'tensors': len(saved['state_dict']),
            'parameters': sum(p.numel() for p in self.tracker.CAMEL.parameters()),
            'missing_keys': [], 'unexpected_keys': []}
        self.tracker.input_columns = list(dict.fromkeys(self.tracker.input_columns))
        self.tracker.device = device
        self.tracker.reset()
        del saved

        saved = torch.load(KPR_CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
        self.cfg = saved['config'].clone()
        self.cfg.defrost()
        self.cfg.model.pretrained = False  # complete checkpoint includes the backbone
        self.cfg.use_gpu = False  # avoid upstream DataParallel/implicit cuda selection
        state = {k.removeprefix('module.'): v for k, v in saved['state_dict'].items()}
        num_classes = state['global_identity_classifier.classifier.weight'].shape[0]
        self.appearance = torchreid.models.build_model(
            name=self.cfg.model.name, num_classes=num_classes, loss=self.cfg.loss.name,
            pretrained=False, use_gpu=False, config=self.cfg,
        )
        self.appearance.load_state_dict(state, strict=True)
        self.loading['appearance'] = {'tensors': len(state),
            'parameters': sum(p.numel() for p in self.appearance.parameters()),
            'missing_keys': [], 'unexpected_keys': []}
        self.appearance.eval().to(device)
        del saved, state
        if self.cfg.model.kpr.keypoints.enabled or self.cfg.model.promptable_trans.masks_prompting:
            raise RuntimeError('This adapter requires the released unprompted appearance checkpoint')
        _, self.transform, _, _ = build_transforms(
            self.cfg.data.height, self.cfg.data.width, self.cfg,
            transforms=self.cfg.data.transforms,
            norm_mean=self.cfg.data.norm_mean, norm_std=self.cfg.data.norm_std,
            remove_background_mask=False, masks_preprocess=self.cfg.model.kpr.masks.preprocess,
            verbose=False,
        )
        self.pose = RTMPose(str(POSE_CHECKPOINT), model_input_size=(192, 256),
                           backend='onnxruntime', device=device)
        expected = 'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'
        if self.pose.session.get_providers()[0] != expected:
            raise RuntimeError(f'RTMPose silently fell back: {self.pose.session.get_providers()}')
        self.pose.session.disable_fallback()
        if device == 'cuda':
            assert all(p.is_cuda for p in self.appearance.parameters())
            assert all(p.is_cuda for p in self.tracker.CAMEL.parameters())
        self.reset()

    def reset(self):
        from cameltrack.cameltrack import Tracklet
        self.tracker.reset()
        Tracklet.count = 0
        self.next_detection = 0
        self.timings = {}

    def _sync(self):
        if self.device == 'cuda':
            import torch
            torch.cuda.synchronize()

    def update(self, frame_bgr, boxes_xyxy, confidence):
        import cv2
        import numpy as np
        import pandas as pd
        import torch

        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        confidence = np.asarray(confidence, dtype=np.float32)
        assert len(boxes) == len(confidence)
        self._sync()
        start = perf_counter()
        if len(boxes):
            keypoints, scores = self.pose(frame_bgr, boxes.tolist())
            poses = np.concatenate([keypoints, scores[..., None]], axis=-1)
        else:
            poses = np.empty((0, 17, 3), np.float32)
        self._sync()
        pose_done = perf_counter()
        embeddings, visibility = [], []
        if len(boxes):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            crops = []
            for box in boxes:
                x1, y1, x2, y2 = np.rint(box).astype(int)
                x1, y1 = np.clip([x1, y1], 0, [width-1, height-1])
                x2, y2 = np.clip([x2, y2], [x1+1, y1+1], [width, height])
                crops.append(self.transform(image=rgb[y1:y2, x1:x2])['image'])
            with torch.inference_mode():
                for offset in range(0, len(crops), 16):
                    output = self.appearance(images=torch.stack(crops[offset:offset+16]).to(self.device))
                    emb, vis, _, _ = self.extract_embeddings(output, self.cfg.model.kpr.test_embeddings)
                    embeddings.extend(emb.cpu().numpy())
                    visibility.extend(vis.cpu().numpy())
            assert np.asarray(embeddings).shape == (len(boxes), 6, 128)
            assert np.asarray(visibility).shape == (len(boxes), 6)
            assert np.isfinite(embeddings).all() and np.isfinite(poses).all()
        self._sync()
        features_done = perf_counter()
        ltwh = boxes.copy()
        ltwh[:, 2:] -= ltwh[:, :2]
        indices = range(self.next_detection, self.next_detection + len(boxes))
        detections = pd.DataFrame({'bbox_ltwh': list(ltwh), 'bbox_conf': confidence,
            'keypoints_xyc': list(poses), 'embeddings': list(embeddings),
            'visibility_scores': list(visibility)}, index=indices)
        self.next_detection += len(boxes)
        image_id = self.tracker.frame_count
        metadata = pd.Series({'id': image_id})
        with torch.inference_mode():
            prepared = self.tracker.preprocess(frame_bgr, detections, metadata)
            result = self.tracker.process(prepared, detections, pd.DataFrame([metadata]))
        self._sync()
        finished = perf_counter()
        self.timings = {'pose_seconds': pose_done-start,
                        'appearance_seconds': features_done-pose_done,
                        'association_seconds': finished-features_done}
        if not isinstance(result, pd.DataFrame) or result.empty:
            return np.empty((0,), dtype=int), np.empty((0, 4), dtype=float)
        # Native CAMEL outputs IDs only for the detections that it emits.
        selected = np.asarray([detections.index.get_loc(i) for i in result.index])
        return result.track_id.to_numpy(dtype=int), boxes[selected].astype(float)
