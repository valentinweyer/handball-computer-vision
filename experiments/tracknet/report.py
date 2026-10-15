"""Compare held-out centre localization and render the unseen BHC-FAG clip."""
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
import torch
from experiments.tracknet.experiment import ROOT, OUT, W, H, make_model, metrics, choose_threshold, write_json


def compare():
    detector=json.loads((OUT/'detector_predictions.json').read_text())
    threshold=choose_threshold([r for r in detector if r['split']=='valid'])['threshold']
    det=[r for r in detector if r['split']=='test']
    temporal=json.loads((OUT/'test_predictions.json').read_text())
    tn_threshold=json.loads((OUT/'test_metrics.json').read_text())['threshold']
    det_by_key={r['key']:r for r in det}
    counts={'both':0,'detector_only':0,'tracknet_only':0,'neither':0}
    for r in temporal:
        if r['xy'] is None: continue
        d=det_by_key[r['key']]
        good_d=d['score']>=threshold and d['distance_input_px'] is not None and d['distance_input_px']<=10
        good_t=r['score']>=tn_threshold and r['distance_input_px']<=10
        counts['both' if good_d and good_t else 'detector_only' if good_d else 'tracknet_only' if good_t else 'neither']+=1
    summary={'detector':metrics(det,threshold),'tracknet':metrics(temporal,tn_threshold),
             'positive_frame_overlap':counts,'note':'Each threshold selected on validation only. Overlap counts are diagnostic, not a deployable fusion result.'}
    write_json(OUT/'comparison.json',summary)
    print(json.dumps(summary,indent=2))
    # Deterministic contact sheet: first 12 held-out positive targets, sorted by source key.
    panels=[]
    for r in sorted((r for r in temporal if r['xy'] is not None),key=lambda r:r['key'])[:12]:
        source=ROOT.parent/'rfdetr-handball-finetune/datasets/expansion-v1/full'/f"{r['key']}.png"
        image=cv2.imread(str(source)); height,width=image.shape[:2]
        tx,ty=r['xy'][0]*width/W,r['xy'][1]*height/H
        cv2.circle(image,(round(tx),round(ty)),16,(0,255,0),2)
        d=det_by_key[r['key']]
        for row,t,c in [(r,tn_threshold,(255,0,255)),(d,threshold,(255,255,0))]:
            if row['score']>=t and row['predicted_xy'] is not None:
                x,y=row['predicted_xy']; cv2.circle(image,(round(x*width/W),round(y*height/H)),23,c,2)
        left=max(0,min(width-240,round(tx)-120)); top=max(0,min(height-180,round(ty)-90))
        panel=cv2.resize(image[top:top+180,left:left+240],(320,240))
        cv2.putText(panel,f"f{r['frame']} TN {r['score']:.2f}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        panels.append(panel)
    cv2.imwrite(str(OUT/'test_contact_sheet.jpg'),np.vstack([np.hstack(panels[i:i+4]) for i in range(0,12,4)]))
    return threshold


def render(detector_threshold):
    checkpoint=torch.load(OUT/'train/best.pt',map_location='cpu',weights_only=True)
    model=make_model(pretrained=False);model.load_state_dict(checkpoint['model']);model.eval()
    threshold=checkpoint['threshold']
    directory=ROOT/'data/cache/frames/BHC-FAG_window_cached'
    files=sorted(directory.glob('*.jpg'),key=lambda p:int(p.stem))[:500]
    cache=np.load(ROOT/'outputs/ball_cache/.BHC-FAG_window_cached_ball_v1.npz')
    detector_threshold=max(detector_threshold, float(cache['threshold']))
    write_json(OUT/'video_settings.json', {'fps':50, 'tracknet_threshold':threshold, 'detector_threshold':detector_threshold, 'detector_cache_floor':float(cache['threshold'])})
    writer=cv2.VideoWriter(str(OUT/'BHC-FAG_comparison.raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50,(960,540))
    if not writer.isOpened(): raise RuntimeError('Could not open video writer')
    predictions=[]; previous=[]
    with torch.inference_mode():
        for index,path in enumerate(files):
            image=cv2.imread(str(path)); h,w=image.shape[:2]
            previous.append(cv2.resize(image,(W,H))); previous=previous[-3:]
            if len(previous)==3:
                array=np.concatenate(previous[::-1],axis=2).transpose(2,0,1).copy()
                inp=torch.from_numpy(array[None]).cuda().float()/255.
                with torch.autocast('cuda',dtype=torch.bfloat16): output=model(inp)
                score,pos=output.float().sigmoid().reshape(-1).max(dim=0)
                score=float(score);pos=int(pos)
                xy=[(pos%W)*w/W,(pos//W)*h/H]
                predictions.append({'frame':index,'xy':xy,'score':score,'accepted':score>=threshold})
                if score>=threshold:
                    cv2.circle(image,tuple(round(v) for v in xy),24,(255,0,255),3)
            for j in range(int(cache['offsets'][index]),int(cache['offsets'][index+1])):
                if cache['confidence'][j]>=detector_threshold:
                    x1,y1,x2,y2=cache['boxes'][j]
                    cv2.circle(image,(round(float((x1+x2)/2)),round(float((y1+y2)/2))),20,(255,255,0),3)
            image=cv2.resize(image,(960,540))
            cv2.putText(image,f'frame {index} | magenta: TrackNet | cyan: RF-DETR',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            writer.write(image)
    writer.release();write_json(OUT/'BHC-FAG_predictions.json',predictions)
    from handball_cv.tracking.sam2_driver import ffmpeg_exe
    subprocess.run([ffmpeg_exe(),'-y','-loglevel','error','-i',str(OUT/'BHC-FAG_comparison.raw.mp4'),'-c:v','libx264','-pix_fmt','yuv420p',str(OUT/'BHC-FAG_comparison.mp4')],check=True)

if __name__=='__main__':
    torch.set_num_threads(4);cv2.setNumThreads(2)
    render(compare())
