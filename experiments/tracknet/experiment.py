"""Isolated temporal handball fine-tuning: prepare, smoke, train, evaluate."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from experiments.tracknet.model import BallTrackerNet

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / 'rfdetr-handball-finetune/datasets/expansion-v1'
WEIGHTS = ROOT / 'models/tracknet-v1-tennis/tracknet_weights.pth'

# Input size, settable as TRACKNET_INPUT=1280x720. At 640x360 a 23 px ball in a
# 1920x1080 source reaches the network 7.7 px wide -- smaller than RF-DETR's
# 8.4 px, and the encoder halves three times before the bottleneck, so roughly
# one pixel survives. Doubling each side is the cheapest test of whether the
# first run measured the architecture or the resize.
#
# CACHE and OUT carry the size because `prepare` skips any .npy that already
# exists: sharing one cache path across sizes would silently mix 640x360 arrays
# into a 1280x720 dataset. The default keeps the original unsuffixed paths, so
# the 2.1 GB cache and the completed run stay exactly where they were.
W, H = (int(value) for value in os.getenv('TRACKNET_INPUT', '640x360').split('x'))
_SUFFIX = '' if (W, H) == (640, 360) else f'-{W}x{H}'
CACHE = ROOT / f'data/cache/tracknet-handball-v1{_SUFFIX}'
OUT = ROOT / f'runs/tracknet_handball_v1{_SUFFIX}'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    matches = json.loads((SOURCE / 'matches.json').read_text())['matches']
    labels = json.loads((SOURCE / 'ball-labels.json').read_text())['labels']
    records = []
    for match in matches:
        capture = cv2.VideoCapture(match['path'])
        if not capture.isOpened():
            raise RuntimeError(f"Cannot read {match['path']}")
        keys = sorted(k for k in labels if k.rsplit('-f', 1)[0] == match['id'])
        for key in keys:
            fid = int(key.rsplit('-f', 1)[1])
            if fid < 2:
                raise ValueError(f'Insufficient context: {key}')
            dest = CACHE / f'{key}.npy'
            record = labels[key]
            if record['status'] not in ('ball', 'none'):
                raise ValueError(record)
            # Check every target against the labelled PNG; metadata alone is insufficient.
            capture.set(cv2.CAP_PROP_POS_FRAMES, fid - 2)
            frames = []
            for index in range(fid - 2, fid + 1):
                ok, image = capture.read()
                if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1 != index:
                    raise RuntimeError(f'Frame decode mismatch: {key}, {index}')
                frames.append(image)
            labelled = cv2.imread(str(SOURCE / 'full' / f'{key}.png'))
            offset = 0
            if labelled is None:
                raise RuntimeError(f'Missing labelled image: {key}')
            if not np.array_equal(labelled, frames[-1]):
                # Some variable-rate sources seek inconsistently. Identify the exact
                # labelled image by pixels in a continuous decode, never by a guessed offset.
                begin = max(0, fid - 8)
                capture.set(cv2.CAP_PROP_POS_FRAMES, begin)
                window = []
                found = False
                for index in range(begin, fid + 9):
                    ok, image = capture.read()
                    if not ok: break
                    window.append(image)
                    if len(window) >= 3 and np.array_equal(labelled, image):
                        frames = window[-3:]
                        offset = index - fid
                        found = True
                        break
                if not found:
                    raise RuntimeError(f'No exact labelled image near source frame: {key}')
            if not dest.exists():
                # Preserve upstream BGR and newest-to-oldest order; no temporal duplication.
                np.save(dest, np.concatenate([cv2.resize(im, (W, H)) for im in frames[::-1]], axis=2))
            records.append({'key': key, 'match': match['id'], 'split': match['split'],
                            'frame': fid, 'decode_offset': offset, 'source': match['path'], 'fps': match['fps'],
                            'width': match['width'], 'height': match['height'],
                            'xy': [record['x'] * W, record['y'] * H] if record['status'] == 'ball' else None})
        capture.release()
        print(f"Prepared and verified {match['id']}: {len(keys)}", flush=True)
    if len(records) != len(labels):
        raise ValueError('Some labels have no match mapping')
    sets = {split: {r['match'] for r in records if r['split'] == split} for split in ('train', 'valid', 'test')}
    assert not sets['train'] & sets['valid'] and not sets['train'] & sets['test'] and not sets['valid'] & sets['test']
    write_json(CACHE / 'manifest.json', {'records': records, 'target_verification': 'all pixel-exact',
               'labels_sha256': hashlib.sha256((SOURCE / 'ball-labels.json').read_bytes()).hexdigest(),
               'matches_sha256': hashlib.sha256((SOURCE / 'matches.json').read_bytes()).hexdigest()})
    print(f'Prepared {len(records)} samples', flush=True)


def target_heatmap(xy):
    target = np.zeros((H, W), np.float32)
    if xy is not None:
        x, y = np.clip(np.rint(xy).astype(int), [0, 0], [W-1, H-1])
        yy, xx = np.ogrid[:H, :W]
        target = np.exp(-((xx-x)**2 + (yy-y)**2) / 8.).astype(np.float32)
    return target


class Frames(Dataset):
    def __init__(self, split, augment=False, limit=None):
        self.records = [r for r in json.loads((CACHE/'manifest.json').read_text())['records'] if r['split'] == split]
        if limit:
            # Include different matches rather than only the first match.
            self.records = random.Random(42).sample(self.records, min(limit, len(self.records)))
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        array = np.load(CACHE / f"{r['key']}.npy")
        target = target_heatmap(r['xy'])
        if self.augment and random.random() < .5:
            array, target = array[:, ::-1], target[:, ::-1]
        # Same geometric transform for all frames and target.
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).float() / 255.
        return tensor, torch.from_numpy(target.copy()), index


def make_model(pretrained=True):
    model = BallTrackerNet()
    if pretrained:
        model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    # Transfer all 17 feature blocks; replace 256 intensity classes by one centre heatmap.
    model.conv18 = nn.Conv2d(64, 1, 1)
    nn.init.normal_(model.conv18.weight, std=.001)
    nn.init.constant_(model.conv18.bias, -4.6)
    model.out_channels = 1
    return model.cuda()


def heatmap_loss(logits, targets):
    # Gaussian focal objective (CenterNet formulation); negatives have an all-zero target.
    p = logits.float().reshape(-1, H, W).sigmoid().clamp(1e-5, 1-1e-5)
    positive = targets.eq(1)
    pos = -p.log() * (1-p).square() * positive
    neg = -(1-p).log() * p.square() * (1-targets).pow(4) * (~positive)
    return (pos.sum() + neg.sum()) / positive.sum().clamp_min(1)


def loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2,
                      pin_memory=True, persistent_workers=True)


@torch.inference_mode()
def predict(model, data_loader):
    model.eval()
    rows = []
    for images, _, indices in data_loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images.cuda(non_blocking=True)).reshape(-1, H*W)
        scores, positions = logits.float().sigmoid().max(dim=1)
        for idx, score, position in zip(indices.tolist(), scores.tolist(), positions.tolist()):
            r = data_loader.dataset.records[idx]
            xy = [position % W, position // W]
            distance = None if r['xy'] is None else float(np.linalg.norm(np.subtract(xy, r['xy'])))
            rows.append({**r, 'predicted_xy': xy, 'score': score, 'distance_input_px': distance})
    return rows


def metrics(rows, threshold, radius=10.):
    present = sum(r['xy'] is not None for r in rows)
    emitted = [r for r in rows if r['score'] >= threshold]
    tp = sum(r['distance_input_px'] is not None and r['distance_input_px'] <= radius for r in emitted)
    fp, fn = len(emitted)-tp, present-tp
    precision = tp/max(1, len(emitted)); recall = tp/max(1, present)
    return {'threshold': float(threshold), 'radius_input_px': radius, 'frames': len(rows),
            'positive_frames': present, 'tp': tp, 'fp': fp, 'fn': fn,
            'false_positives_on_absent': sum(r['xy'] is None for r in emitted),
            'precision': precision, 'recall': recall, 'f1': 2*tp/max(1, 2*tp+fp+fn)}


def choose_threshold(rows):
    # Predetermined grid, selected only on validation.
    return max((metrics(rows, float(t)) for t in [.01,.02,.03,.04]+list(np.arange(.05,1,.05))), key=lambda m: (m['f1'],m['precision']))


def train(args):
    run = OUT / ('smoke' if args.smoke else 'train')
    run.mkdir(parents=True, exist_ok=True)
    if (run/'best.pt').exists():
        raise FileExistsError(f'Existing run: {run}')
    train_data = Frames('train', augment=not args.smoke, limit=12 if args.smoke else None)
    val_data = train_data if args.smoke else Frames('valid')
    train_loader = loader(train_data, args.batch_size, shuffle=True)
    val_loader = loader(val_data, args.batch_size)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    epochs = 12 if args.smoke else args.epochs
    config = {'seed':42, 'batch_size':args.batch_size, 'lr':args.lr,'max_epochs':epochs,
              'patience':args.patience,'pretrained':str(WEIGHTS),'head':'single-channel sigmoid heatmap',
              'loss':'Gaussian focal; sigma 2 input pixels','input':'BGR t,t-1,t-2 /255; 640x360',
              'batchnorm':'pretrained running statistics frozen; affine parameters trainable',
              'augmentation':'none' if args.smoke else 'synchronized horizontal flip',
              'checkpoint_sha256':hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
              'manifest_sha256':hashlib.sha256((CACHE/'manifest.json').read_bytes()).hexdigest()}
    write_json(run/'config.json', config)
    history=[]; best=-1.; stale=0
    for epoch in range(1, epochs+1):
        started=time.perf_counter(); model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
        losses=[]
        for images, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output=model(images.cuda(non_blocking=True))
            loss=heatmap_loss(output, targets.cuda(non_blocking=True))
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss: {loss}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step(); losses.append(float(loss.detach()))
        predictions=predict(model,val_loader); measured=choose_threshold(predictions)
        row={'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.perf_counter()-started,**measured}
        history.append(row); write_json(run/'history.json',history)
        print(json.dumps(row),flush=True)
        if measured['f1'] > best:
            best=measured['f1']; stale=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'threshold':measured['threshold'],'config':config},run/'best.pt')
            write_json(run/'validation_predictions.json',predictions)
        else: stale+=1
        if not args.smoke and epoch>=10 and stale>=args.patience: break
    print(f'Finished {run}; best validation F1 {best:.4f}',flush=True)


def evaluate(args):
    checkpoint=torch.load(OUT/'train/best.pt',map_location='cpu',weights_only=True)
    model=make_model(pretrained=False); model.load_state_dict(checkpoint['model'],strict=True)
    rows=predict(model,loader(Frames('test'),args.batch_size))
    summary=metrics(rows,checkpoint['threshold'])
    summary['epoch']=checkpoint['epoch']
    summary['per_match']={m:metrics([r for r in rows if r['match']==m],checkpoint['threshold']) for m in sorted({r['match'] for r in rows})}
    write_json(OUT/'test_predictions.json',rows); write_json(OUT/'test_metrics.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','evaluate'])
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--patience',type=int,default=7)
    parser.add_argument('--lr',type=float,default=1e-4)
    args=parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4); cv2.setNumThreads(2)
    if args.action=='prepare': prepare()
    elif args.action=='train': train(args)
    else: evaluate(args)

if __name__=='__main__': main()
