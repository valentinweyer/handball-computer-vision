"""Fixed twelve-event comparison. CPU SORT; replay existing ARC and SAM2."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
import cv2
import numpy as np
import pandas as pd
from experiments.box_tracker_oracle.appearance_sort import AppearanceSORT
from experiments.box_tracker_oracle.common import colour_feature
from experiments.arc_team_repair.repair import parse_lines
from scripts.render_raw_team_classification import load_detection_cache, person_detections
from scripts.evaluate_tracker_identity import iou_matrix
from handball_cv.video import ffmpeg_exe

MAIN = Path('/home/valentinweyer/projects/handball-computer-vision')
OLD = MAIN / 'runs/arc-tenmin-review-2026-10-22'
METHODS = ['arc', 'colour_sort', 'sam2_pipeline']
COLORS = [(0, 220, 255), (255, 150, 30)]

def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def read_sam(path):
    df = pd.read_parquet(path, columns=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])
    assert not df.duplicated(['frame', 'track_id']).any()
    return {int(f): {int(row[0]): np.asarray(row[1:], float) for row in group[['track_id','x1','y1','x2','y2']].to_numpy()}
            for f, group in df.groupby('frame')}

def run_sort(root, protocol):
    path = root / 'colour_sort.txt'
    if path.exists(): return parse_lines(path.read_text().splitlines())
    cache = load_detection_cache(OLD / 'detections.npz')
    assert len(cache['offsets']) == 15001
    # Reuse unchanged experiment association, replacing its current-frame feature slot.
    features = [None] * 15000
    tracker = AppearanceSORT(features, 'gate', .28, frame_rate=25,
                             lost_track_buffer=30, track_activation_threshold=.5)
    cap = cv2.VideoCapture(protocol['inputs']['video']['path'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    lines=[]
    try:
        for f in range(15000):
            ok, bgr = cap.read()
            if not ok: raise RuntimeError(f'video decode failed at {f}')
            det = person_detections(cache, f)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            features[f] = np.stack([colour_feature(rgb, b) for b in det.xyxy]) if len(det) else np.empty((0,128),np.float32)
            tracked = tracker.update(det)
            for tid, box in zip(tracked.tracker_id, tracked.xyxy):
                x1,y1,x2,y2 = box
                lines.append(f'{f+1},{int(tid)},{x1:.6f},{y1:.6f},{x2-x1:.6f},{y2-y1:.6f},1,-1,-1,-1')
            features[f]=None
            if f%500 == 0:
                write(root/'status.json', {'stage':'colour_sort','frames':f+1,'total':15000})
                print('SORT', f+1, flush=True)
        assert not cap.read()[0], 'video has unexpected extra frames'
    finally: cap.release()
    temp=path.with_suffix('.partial');temp.write_text('\n'.join(lines)+'\n');temp.replace(path)
    return parse_lines(lines)

def anchor(item, arc):
    # Choose a clear pre-overlap frame using ARC geometry only; no verdicts consulted.
    candidates=[]
    for f in range(item['start'], item['event']['start']):
        rows=arc.get(f,{})
        if not all(t in rows for t in item['focus']): continue
        boxes=np.stack([rows[t] for t in item['focus']])
        candidates.append((float(iou_matrix(boxes[:1],boxes[1:])[0,0]),f))
    return min(candidates)[1] if candidates else item['start']

def map_focus(boxes, rows):
    ids=list(rows)
    if not ids: return [None,None]
    overlaps=iou_matrix(boxes,np.stack(list(rows.values())))
    result=[]
    for scores in overlaps:
        order=np.argsort(-scores)
        good=scores[order[0]]>=.5 and (len(order)<2 or scores[order[0]]-scores[order[1]]>=.1)
        result.append(ids[order[0]] if good else None)
    if result[0] is not None and result[0]==result[1]: return [None,None]
    return result

def annotate(bgr, rows, focus, title):
    pane=bgr.copy()
    for tid,box in rows.items():
        xy=np.rint(box).astype(int)
        index=focus.index(tid) if tid in focus else None
        col=COLORS[index] if index is not None else (160,160,160)
        label=(('X' if index==0 else 'Y')+' / ' if index is not None else '')+f'T{tid}'
        cv2.rectangle(pane,tuple(xy[:2]),tuple(xy[2:]),col,4 if index is not None else 1)
        point=(max(0,xy[0]),max(55,xy[1]))
        cv2.putText(pane,label,point,cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),4)
        cv2.putText(pane,label,point,cv2.FONT_HERSHEY_SIMPLEX,.7,col,2)
    cv2.rectangle(pane,(0,0),(1920,42),(0,0,0),-1)
    cv2.putText(pane,title,(15,29),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    return pane

def render(root,p,paths):
    review=root/'review';review.mkdir(exist_ok=True)
    items=json.loads((OLD/'review/manifest.json').read_text())['items']
    public=[]
    cap=cv2.VideoCapture(p['inputs']['video']['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    try:
        for item in items:
            f=anchor(item,paths['arc']); source=paths['arc'].get(f,{})
            if not all(t in source for t in item['focus']): raise RuntimeError('no shared anchor '+item['id'])
            boxes=np.stack([source[t] for t in item['focus']])
            mapped={m:map_focus(boxes,paths[m].get(f,{})) for m in METHODS}
            assert mapped['arc']==item['focus']
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
            cv2.imwrite(str(review/(item['id']+'-anchor.jpg')),annotate(bgr,dict(zip(item['focus'],boxes)),item['focus'],f'Targets X / Y before overlap | {f/25:.2f}s'))
            # Deterministic label rotation avoids always putting one method first.
            order=METHODS[len(public)%3:]+METHODS[:len(public)%3]
            public_item={'id':item['id'],'start':item['start'],'end':item['end'],'anchor':f,
                         'anchor_image':item['id']+'-anchor.jpg','panels':[]}
            processes=[]
            try:
                for idx,m in enumerate(order):
                    label='ABC'[idx];dest=review/f'{item["id"]}-{label}.mp4';temp=dest.with_suffix('.partial.mp4')
                    cmd=[ffmpeg_exe(),'-nostdin','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(temp)]
                    log=(root/'render-errors.log').open('ab')
                    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
                    processes.append((m,label,proc,log,temp,dest))
                    public_item['panels'].append({'label':label,'video':dest.name,'anchor_ids':mapped[m]})
                cap.set(cv2.CAP_PROP_POS_FRAMES,item['start'])
                for frame in range(item['start'],item['end']+1):
                    ok,bgr=cap.read()
                    if not ok: raise RuntimeError(f'decode failed at {frame}')
                    for m,label,proc,log,temp,dest in processes:
                        pane=annotate(bgr,paths[m].get(frame,{}),mapped[m],f'{label} | {frame/25:.2f}s | fixed local IDs; no number links')
                        proc.stdin.write(pane.tobytes())
            finally:
                for m,label,proc,log,temp,dest in processes:
                    proc.stdin.close();result=proc.wait();log.close()
                    if result: raise RuntimeError(f'encoder failed for {dest}')
                    temp.replace(dest)
            public.append(public_item)
            write(root/'status.json',{'stage':'render','cases':len(public),'total':len(items)})
            print('rendered',item['id'],flush=True)
    finally: cap.release()
    write(review/'manifest.json',{'items':public,'scope':'12 ARC-selected dev overlap events; not overall accuracy','ground_truth':False})
    write(root/'method-key.json',{item['id']:dict(zip('ABC',METHODS[i%3:]+METHODS[:i%3])) for i,item in enumerate(items)})
    template=(root/'review-template.html').read_text()
    (review/'index.html').write_text(template.replace('__ITEMS__',json.dumps(public)))
    (review/'READ-ME.txt').write_text('Open index.html. Check X/Y in the anchor, then judge EACH method independently. A/B/C rotate across cases; method-key.json is kept outside review. New/grey IDs after an overlap can indicate fragmentation. Missing anchor IDs require manual inspection, not automatic failure. Exclude bench/off-court players. Export JSON. These are selected development events, not overall tracking accuracy.\n')

def verify(root,p):
    items=json.loads((root/'review/manifest.json').read_text())['items'];checks=[]
    for item in items:
        for panel in item['panels']:
            path=root/'review'/panel['video'];cap=cv2.VideoCapture(str(path));n=0
            assert cap.isOpened()
            while True:
                ok,frame=cap.read()
                if not ok: break
                assert frame.shape[:2]==(1080,1920);n+=1
            cap.release();assert n==item['end']-item['start']+1,(path,n)
            checks.append({'file':panel['video'],'frames':n,'sha256':sha(path)})
    for value in p['inputs'].values(): assert sha(Path(value['path']))==value['sha256']
    write(root/'verification.json',{'all_videos_fully_decoded':True,'inputs_unchanged':True,'videos':checks})
    write(root/'complete.json',{'complete':True,'cases':len(items),'videos':len(checks),'ground_truth':False,'accuracy':None})
    write(root/'status.json',{'stage':'complete'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    cv2.setNumThreads(1)
    p=json.loads((root/'protocol.json').read_text())
    try:
        for value in p['inputs'].values(): assert sha(Path(value['path']))==value['sha256']
        paths={'arc':parse_lines((OLD/'baseline.txt').read_text().splitlines()),
               'sam2_pipeline':read_sam(Path(p['inputs']['sam2_tracks']['path']))}
        paths['colour_sort']=run_sort(root,p)
        for m,byframe in paths.items():
            assert all(0<=f<15000 for f in byframe),m
            assert all(len(rows)==len(set(rows)) for rows in byframe.values())
        render(root,p,paths);verify(root,p)
    except Exception:
        write(root/'status.json',{'stage':'failed','error':traceback.format_exc()});raise
if __name__=='__main__':main()
