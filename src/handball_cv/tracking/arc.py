"""Pinned ARC-SORT backend. Native association plus local OSNet inference.

No custom repair, merging, interpolation, team gating, or network downloads.
The external checkout and weights are explicit assets; importing this module
never imports ARC or initializes torch/CUDA.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

UPSTREAM_COMMIT = '2a580c592c838bce99b89a6a2dbd7ad6f50c84be'
WEIGHTS_SHA256 = '8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd'


def require_dense_cache(cache: dict, frames: int) -> None:
    """Empty frames are legal; absent evidence of every-frame scoring is not."""
    if (int(cache.get('stride', 0)) != 1
            or int(cache.get('scored_frames', -1)) != frames
            or int(cache.get('total_frames', -1)) != frames
            or len(cache['offsets']) != frames + 1):
        raise ValueError('ARC requires a verified stride-1 detection cache covering every frame; '
                         'use a fresh work directory or --force scan')


def load_arc_cache(path: Path, frames: int) -> dict:
    # The legacy diagnostic loader intentionally strips stride/provenance.
    # ARC must validate the original file rather than trusting that view.
    with np.load(path, allow_pickle=False) as data:
        cache = {k: data[k] for k in data.files}
    require_dense_cache(cache, frames)
    return cache


def assets(source: Path, weights: Path) -> dict:
    source, weights = Path(source).resolve(), Path(weights).resolve()
    if not (source / 'arcsort/tracker.py').is_file() or not weights.is_file():
        raise FileNotFoundError('ARC needs --arc-root (pinned checkout) and --arc-weights (sports OSNet); '
                                'see docs/arc-pipeline.md')
    revision = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f'ARC checkout must be pinned to {UPSTREAM_COMMIT}, got {revision}')
    clean = subprocess.run(['git', '-C', str(source), 'diff', 'HEAD', '--exit-code', '--',
                            'arcsort', 'third_party/osnet/osnet.py'], capture_output=True)
    if clean.returncode:
        raise ValueError('ARC source/config differs from the pinned checkout')
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError('ARC sports OSNet weight checksum differs from the evaluated model')
    return {'upstream_commit': revision, 'weights_sha256': digest,
            'track_high_thresh': .5, 'new_track_thresh': .5,
            'offline_merge': False, 'interpolation': False, 'team_gating': False}


def import_arc(source: Path):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    from arcsort import ARCTracker
    from arcsort.tracker import load_config
    import arcsort
    if source not in Path(arcsort.__file__).resolve().parents:
        raise RuntimeError('A different ARC installation was already imported')
    return ARCTracker, load_config


def box_iou(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                    - np.maximum(a[:, None, :2], b[None, :, :2]))
    intersection = wh.prod(axis=2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b - intersection, 1e-9)


def number_box_pairs(player_boxes, detection_boxes, number_boxes):
    """Own a number from raw detections first; ambiguous overlaps abstain.

    Then map its unique raw owner to one uniquely supported tracked box. A
    competing unconfirmed detection can veto ownership too. Boxes are never
    presented as segmentation masks. Caller retains existing readability/votes.
    """
    players = np.asarray(player_boxes).reshape(-1, 4)
    detections = np.asarray(detection_boxes).reshape(-1, 4)
    numbers = np.asarray(number_boxes).reshape(-1, 4)
    if not len(players) or not len(detections) or not len(numbers): return []
    wh = np.maximum(0, np.minimum(detections[:, None, 2:], numbers[None, :, 2:])
                    - np.maximum(detections[:, None, :2], numbers[None, :, :2]))
    area = np.maximum(0, numbers[:, 2:] - numbers[:, :2]).prod(axis=1)
    owners = wh.prod(axis=2) / np.maximum(area, 1e-9) > .5
    ious = box_iou(players, detections)
    pairs = []
    for n in range(len(numbers)):
        candidates = np.flatnonzero(owners[:, n])
        if len(candidates) != 1: continue
        scores = ious[:, candidates[0]]
        order = np.argsort(-scores)
        if scores[order[0]] < .5: continue
        if len(order) > 1 and scores[order[0]] - scores[order[1]] < .1: continue
        pairs.append((int(order[0]), n))
    return pairs


class ArcTracker:
    """Supervision adapter with explicit OSNet device; association stays on CPU."""
    def __init__(self, source: Path, weights: Path, device='cpu'):
        self.provenance = {**assets(source, weights), 'appearance_device': device}
        self.factory, load_config = import_arc(source)
        self.config = load_config(Path(source) / 'arcsort/configs/sportsmot.yaml')
        self.config.update(track_high_thresh=.5, new_track_thresh=.5)
        import torch
        self.torch, self.device = torch, torch.device(device)
        spec = importlib.util.spec_from_file_location('_handball_arc_osnet',
                                                     Path(source) / 'third_party/osnet/osnet.py')
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict', checkpoint)
        state = {k.removeprefix('module.'): v for k, v in state.items() if torch.is_tensor(v)}
        self.net = mod.osnet_x1_0(num_classes=state['classifier.weight'].shape[0],
                                  pretrained=False, loss='softmax').eval()
        self.net.load_state_dict(state, strict=True)
        self.net.to(self.device)
        self.offset = 0
        self.highest_id = 0
        self.native = self._new_native()

    def _new_native(self):
        native = self.factory(self.config)
        native.embedding_computer._emb_dim = 512
        def missing(*args, **kwargs):
            raise RuntimeError('ARC feature-cache miss; implicit upstream CUDA fallback prohibited')
        native.embedding_computer._compute_from_frame = missing
        return native

    @property
    def alive_ids(self):
        return [int(t.track_id) + 1 + self.offset for t in self.native.tracks]

    def reset(self):
        self.offset = self.highest_id
        self.native = self._new_native()

    def _encode(self, rgb, boxes):
        torch = self.torch
        batch=[]; height,width=rgb.shape[:2]
        for box in boxes:
            x1,y1,x2,y2=np.round(box).astype(int)
            x1,x2=np.clip([x1,x2],0,width); y1,y2=np.clip([y1,y2],0,height)
            crop=rgb[y1:y2,x1:x2]
            patch=cv2.resize(crop,(128,256)) if crop.size else np.zeros((256,128,3),np.uint8)
            batch.append(patch.transpose(2,0,1).astype(np.float32))
        if not batch: return np.empty((0,512),np.float32)
        mean=torch.tensor([.485,.456,.406],device=self.device).view(1,3,1,1)
        std=torch.tensor([.229,.224,.225],device=self.device).view(1,3,1,1)
        result=[]
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
            for start in range(0,len(batch),16):
                x=torch.from_numpy(np.stack(batch[start:start+16])).to(self.device)/255.
                result.append(torch.nn.functional.normalize(self.net((x-mean)/std),dim=1).cpu().numpy())
        return np.concatenate(result)

    def update(self, detections, frame):
        data=np.empty((len(detections),6),float)
        data[:,:4]=detections.xyxy
        data[:,4]=detections.confidence
        data[:,5]=detections.class_id
        self.native.embedding_computer._cache={self.native.frame_count+1:
            (detections.xyxy.astype(float), self._encode(frame,detections.xyxy))}
        rows=self.native.update(data,cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)).reshape(-1,5)
        ids=rows[:,4].astype(int)+self.offset
        self.highest_id=max([self.highest_id,*self.alive_ids])
        classes=np.full(len(rows),-1,int); confidence=np.zeros(len(rows))
        if len(rows) and len(detections):
            scores=box_iou(rows[:,:4],detections.xyxy)
            ri,di=linear_sum_assignment(-scores)
            for row,col in zip(ri,di):
                if scores[row,col]>=.5:
                    classes[row]=detections.class_id[col];confidence[row]=detections.confidence[col]
        return sv.Detections(xyxy=rows[:,:4],tracker_id=ids,class_id=classes,confidence=confidence)


@dataclass
class ArcFrame:
    frame_idx: int
    player_ids: np.ndarray
    boxes: np.ndarray
    masks: list
    rgb: np.ndarray
    detection_boxes: np.ndarray
    def read_frame(self): return self.rgb


class ArcDriver:
    """Keep native ARC IDs separate from the shared registry's identity IDs."""
    def __init__(self, tracker, identity):
        self.tracker, self.identity = tracker, identity
        self.registry = identity.registry
        self.events = []
        self.local_rows = []

    def frames(self, source, detections_for, court_in_view=None):
        from handball_cv.video import VideoFrames
        video=VideoFrames(source)
        from tqdm import tqdm
        paused=False
        arbitration_count=0
        for f in tqdm(range(len(video)), desc="ARC pipeline"):
            rgb=video.rgb(f);det=detections_for(f)
            if court_in_view is not None and not court_in_view(f):
                if not paused:
                    self.events.append({'frame':f,'type':'pause'})
                    self.tracker.reset();paused=True
                    arbitration_count=0
                self.identity.retire_missing(f,[])
                yield ArcFrame(f,np.empty(0,int),np.empty((0,4)),[],rgb,det.xyxy)
                continue
            if paused:
                self.events.append({'frame':f,'type':'resume'});paused=False
            tracked=self.tracker.update(det,rgb)
            native = getattr(self.tracker, "native", None)
            arbitration = getattr(native, "arbitration_swaps", [])
            for event in arbitration[arbitration_count:]:
                self.events.append({"frame":f,"type":"arc_arbitration",
                                    "local_id_offset":self.tracker.offset,"native":event})
            arbitration_count = len(arbitration)
            self.identity.retire_missing(f,self.tracker.alive_ids)
            ids=self.identity.update(f,rgb,tracked)
            self.local_rows.extend((f,int(tid),int(pid)) for tid,pid in zip(tracked.tracker_id,ids))
            yield ArcFrame(f,ids,tracked.xyxy,[None]*len(ids),rgb,det.xyxy)

    def save_local_tracks(self, path):
        rows=np.asarray(self.local_rows,dtype=np.int64).reshape(-1,3)
        np.savez_compressed(path,frame=rows[:,0],local_track_id=rows[:,1],identity_id=rows[:,2])
