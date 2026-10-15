"""Score existing RF-DETR on the same labelled targets, in its own environment.
Run: ../rfdetr-handball-finetune/.venv/bin/python experiments/tracknet/detector_baseline.py
"""
import json
from pathlib import Path
import cv2
import numpy as np
from rfdetr import RFDETRLarge
ROOT=Path(__file__).resolve().parents[2]
source=ROOT.parent/'rfdetr-handball-finetune'
records=json.loads((ROOT/'data/cache/tracknet-handball-v1/manifest.json').read_text())['records']
checkpoint=source/'runs/ball-v1-s42/checkpoint_best_total.pth'
model=RFDETRLarge(pretrain_weights=str(checkpoint))
rows=[]
for r in records:
    if r['split']=='train': continue
    image=cv2.imread(str(source/'datasets/expansion-v1/full'/f"{r['key']}.png"))
    result=model.predict(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),threshold=.01)
    xy=None; score=0.
    if len(result):
        i=int(np.argmax(result.confidence)); box=result.xyxy[i]
        xy=[float((box[0]+box[2])*.5*640/r['width']),float((box[1]+box[3])*.5*360/r['height'])]
        score=float(result.confidence[i])
    distance=None if xy is None or r['xy'] is None else float(np.linalg.norm(np.subtract(xy,r['xy'])))
    rows.append({**r,'predicted_xy':xy,'score':score,'distance_input_px':distance})
    if len(rows)%50==0: print(f'{len(rows)}/500',flush=True)
out=ROOT/'runs/tracknet_handball_v1/detector_predictions.json'
out.write_text(json.dumps(rows,indent=2))
print(out,flush=True)
