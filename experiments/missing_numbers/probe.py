"""Isolated pixel review and OCR probe; never writes input caches or votes."""
from pathlib import Path
from collections import Counter
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import supervision as sv

from handball_cv.teams.model import crop_quality
from handball_cv.jersey.parseq_backend import load_jersey_parseq, read_crops
from handball_cv.jersey.identity import is_valid_number

root = Path('runs/missing-numbers-audit-2026-10-18')
bundle = Path('runs/bundles/Kiel_Lemgo_66m00_10min_expanded_validation_v4')
o = pd.read_parquet(root / 'opportunities.parquet')
b = pd.read_parquet(root / 'number-boxes.parquet').set_index(['frame', 'number_row'])
t = pd.read_parquet(bundle / 'tracks.parquet').set_index(['frame', 'track_id'])
episodes = json.loads((root / 'missing-episodes.json').read_text())
selected, used = [], set()
for stage, quota in [('read_accepted', 8), ('crop_gate', 8), ('ocr_abstain', 6),
                     ('ambiguous_mask_owner', 2), ('mask_overlap_below_threshold', 2),
                     ('no_number_detection_near_player', 4)]:
    candidates = sorted(episodes, key=lambda e: (-e['stages'].get(stage, 0), -e['frames']))
    for e in candidates:
        if e['track_id'] in used or not e['stages'].get(stage):
            continue
        obs = o[(o.track_id == e['track_id']) & o.frame.between(e['first'], e['last']) &
                (o.stage == stage)]
        indices = np.linspace(0, len(obs) - 1, 3).round().astype(int)
        frames = obs.iloc[indices].frame.astype(int).tolist()
        selected.append(dict(case=len(selected) + 1, reason=stage, **e, sample_frames=frames))
        used.add(e['track_id'])
        quota -= 1
        if not quota:
            break
assert len(selected) == 30
controls = []
for row in o[(o.number.notna()) & (o.stage == 'read_accepted')].groupby('player_id').first().head(6).itertuples():
    controls.append(dict(frame=int(row.frame), track_id=int(row.track_id), number=row.number))
(root / 'sample.json').write_text(json.dumps(dict(cases=selected, controls=controls), indent=2) + '\n')

wanted = {}
for case in selected:
    for f in case['sample_frames']:
        wanted.setdefault(f, []).append((case['case'], case['track_id']))
for control in controls:
    wanted.setdefault(control['frame'], []).append((31 + controls.index(control), control['track_id']))
gated = o[o.stage == 'crop_gate'].copy()
gate_by_frame = {int(f): g for f, g in gated.groupby('frame')}
all_frames = sorted(set(wanted) | set(gate_by_frame))
cap = cv2.VideoCapture(str(bundle / 'video.mp4'))
assert cap.isOpened()
crops, gate_records, pixels = [], [], {}
last = -1
for f in all_frames:
    for _ in range(f - last - 1):
        assert cap.grab()
    ok, frame_bgr = cap.read()
    assert ok
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    last = f
    for row in gate_by_frame.get(f, pd.DataFrame()).itertuples():
        box = b.loc[(f, int(row.number_row)), ['x1', 'y1', 'x2', 'y2']].to_numpy(float)
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        quality = crop_quality(frame[y1:y2, x1:x2])
        assert not quality.accepted, (f, row.track_id)
        failures = []
        if min(quality.width, quality.height) < 10: failures.append('side_lt_10')
        if quality.width * quality.height < 400: failures.append('area_lt_400')
        if quality.contrast < 12: failures.append('contrast_lt_12')
        if quality.sharpness < 20: failures.append('sharpness_lt_20')
        # OCR uses supervision's truncation, while the quality gate rounds.
        crop = sv.crop_image(frame, box)
        gate_records.append(dict(frame=f, track_id=int(row.track_id), player_id=int(row.player_id),
                                 number_row=int(row.number_row), display_number=row.number,
                                 width=quality.width, height=quality.height,
                                 contrast=quality.contrast, sharpness=quality.sharpness,
                                 failures=','.join(failures)))
        crops.append(crop)
    for case, track in wanted.get(f, []):
        row = t.loc[(f, track)]
        box = row[['x1', 'y1', 'x2', 'y2']].to_numpy(float)
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, y1, x2, y2 = max(0, x1 - 10), max(0, y1 - 10), min(1920, x2 + 10), min(1080, y2 + 10)
        crop = frame[y1:y2, x1:x2]
        pixels[(case, f)] = Image.fromarray(crop) if crop.size else Image.new('RGB', (100, 200))
        if case <= 30:
            cv2.imwrite(str(root / f'case-{case:02d}-f{f}.jpg'), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
cap.release()
print('All 2136 crop-gate exclusions reproduced from pixels', len(crops), flush=True)
reader = load_jersey_parseq(Path('models/jersey_parseq/parseq_original.ckpt'))
outputs = read_crops(*reader, crops, batch_size=64)
for record, (text, confidence) in zip(gate_records, outputs):
    record.update(probe_text=text, probe_confidence=confidence,
                  probe_accepted=bool(confidence >= .5 and is_valid_number(str(text))))
probe = pd.DataFrame(gate_records)
probe.to_parquet(root / 'crop-gate-probe.parquet', index=False)
summary = dict(gated_crops=len(probe),
               gate_reasons=dict(Counter(probe.failures)),
               probe_valid_reads=int(probe.probe_accepted.sum()),
               missing_crop_gate=int(probe.display_number.isna().sum()),
               missing_probe_valid_reads=int(probe.loc[probe.display_number.isna(), 'probe_accepted'].sum()))
(root / 'probe-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)

font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(font_path, 17)
small = ImageFont.truetype(font_path, 14)
for page in range(5):
    sheet = Image.new('RGB', (1200, 1200), '#eeeeee')
    draw = ImageDraw.Draw(sheet)
    for index, case in enumerate(selected[page*6:(page+1)*6]):
        left, top = (index % 2) * 600, (index // 2) * 400
        draw.text((left+10, top+5), f"Case {case['case']:02d} P{case['player_id']} / track {case['track_id']} "
                  f"{case['first']/25:.1f}-{case['last']/25:.1f}s", font=font, fill='black')
        draw.text((left+10, top+29), case['reason'], font=small, fill='black')
        draw.text((left+10, top+49), f"Accepted reads in episode: {case['reads']}", font=small, fill='black')
        for column, f in enumerate(case['sample_frames']):
            im = pixels[(case['case'], f)].copy()
            im.thumbnail((185, 300))
            x, y = left + column*195 + 8, top + 77
            sheet.paste(im, (x + (185-im.width)//2, y))
            draw.text((x, top+380), f'f{f} / {f/25:.1f}s', font=small, fill='black')
    sheet.save(root / f'review-{page+1}.jpg', quality=92)
sheet = Image.new('RGB', (1200, 400), '#eeeeee')
draw = ImageDraw.Draw(sheet)
for index, case in enumerate(controls):
    im = pixels[(31+index, case['frame'])].copy(); im.thumbnail((190, 320))
    sheet.paste(im, (index*200+(190-im.width)//2, 40))
    draw.text((index*200+5, 5), f"Control #{case['number']}", font=font, fill='black')
    draw.text((index*200+5, 375), f"f{case['frame']}", font=small, fill='black')
sheet.save(root/'controls.jpg', quality=92)
print('Wrote 30 three-frame cases and six controls', flush=True)
