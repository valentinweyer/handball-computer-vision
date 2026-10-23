"""Trace cached number opportunities without rerunning any models.

Run from the repository root with its existing venv. Output is separate from
the immutable input bundle. Stage names describe recorded pipeline evidence,
not human readability or correctness; those require inspecting the images.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from handball_cv.jersey.identity import _ios_box_local, MATCH_IOS_MIN
from handball_cv.tracking.geometry_cache import GeometryCache
from handball_cv.pipeline.tables import _segments_by_player
from scripts.render_full_pipeline import unique_pairs
from scripts.render_raw_team_classification import load_detection_cache, number_detections


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_labels(bundle: Path, work: Path, output: Path):
    """Separate the final duplicate-number guard from earlier abstentions."""
    run = json.loads((work / 'track.json').read_text())
    tracks = pd.read_parquet(bundle / 'tracks.parquet')
    segments = _segments_by_player(run, 5, fold_links=True)
    losses = []
    prospective = []
    for row in tracks.itertuples(index=False):
        segment = next((s for s in segments.get(row.player_id, [])
                        if s.start <= row.frame <= s.end), None)
        value = segment.value_at(row.frame) if segment else None
        prospective.append(value)
        if value is not None and pd.isna(row.number):
            losses.append(dict(frame=int(row.frame), player_id=int(row.player_id),
                               track_id=int(row.track_id), team=int(row.team),
                               prospective=value, live=row.number_live))
    pd.DataFrame(losses).to_csv(output / 'final-duplicate-guard-losses.csv', index=False)
    t = tracks.assign(prospective=prospective)
    snapshots = []
    for frame, value in [(8850, '3'), (12125, '53')]:
        snapshots.extend(t[(t.frame == frame) & (t.prospective == value)][
            ['frame', 'player_id', 'track_id', 'team', 'prospective', 'number',
             'number_live', 'x1', 'y1', 'x2', 'y2']].to_dict('records'))
    summary = dict(withheld_by_final_duplicate_guard=len(losses),
                   missing_before_final_guard=int(pd.isna(prospective).sum()),
                   collision_snapshots=snapshots)
    (output / 'label-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def audit(bundle: Path, work: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    input_names = ['manifest.json', 'tracks.parquet', 'reads.parquet',
                   'players.parquet', 'events.parquet', 'detections.parquet']
    before = {name: digest(bundle / name) for name in input_names}
    tracks = pd.read_parquet(bundle / 'tracks.parquet')
    reads = pd.read_parquet(bundle / 'reads.parquet')
    geometry = GeometryCache.load(bundle / 'masks.npz')
    cache = load_detection_cache(work / 'detections.npz')
    raw = json.loads((work / 'track_reads.json').read_text())
    read_keys = {(int(r.frame), int(r.track_id), int(r.number_row)): str(r.value)
                 for r in reads.itertuples()}
    records = []
    box_records = []
    validated_reads = set()
    counted_attempts = set()
    cadence = manifest['parameters']['ocr_every']
    frames = tracks[tracks.frame % cadence == 0].groupby('frame', sort=True)
    for frame, rows in frames:
        frame = int(frame)
        fg = geometry.frame(frame)
        rows = rows.sort_values('mask_row')
        assert rows.track_id.tolist() == fg.player_ids.tolist()
        boxes = number_detections(cache, frame)
        hit = raw.get(str(frame), {})
        ios = (_ios_box_local(fg.masks, boxes, geometry.width, geometry.height)
               if len(boxes) else np.zeros((len(rows), 0)))
        tied = (np.isclose(ios, ios.max(axis=0), rtol=0, atol=1e-6).sum(axis=0) > 1
                if len(boxes) else np.zeros(0, dtype=bool))
        pr, nr = np.where(ios > MATCH_IOS_MIN)
        pairs = [(int(p), int(n)) for p, n in zip(pr, nr) if not tied[n]]
        pairs.sort(key=lambda pair: ios[pair], reverse=True)
        selected = dict(unique_pairs(pairs))
        for number_row, box in enumerate(boxes):
            box_records.append(dict(frame=frame, number_row=number_row,
                                    x1=float(box[0]), y1=float(box[1]),
                                    x2=float(box[2]), y2=float(box[3]),
                                    top_ios=float(ios[:, number_row].max()),
                                    ambiguous=bool(tied[number_row]),
                                    attempted=str(number_row) in hit,
                                    value=hit.get(str(number_row))))
        for local, row in enumerate(rows.itertuples(index=False)):
            number_row = selected.get(local)
            near = []
            for n, box in enumerate(boxes):
                cx, cy = (box[:2] + box[2:]) / 2
                if row.x1 <= cx <= row.x2 and row.y1 <= cy <= row.y2:
                    near.append(n)
            text = None
            if number_row is not None:
                if str(number_row) not in hit:
                    stage = 'crop_gate'
                else:
                    counted_attempts.add((frame, number_row))
                    text = hit[str(number_row)]
                    stage = 'read_accepted' if text else 'ocr_abstain'
                    if text:
                        key = frame, int(row.track_id), number_row
                        assert read_keys[key] == text, (key, text)
                        validated_reads.add(key)
            elif any(p == local for p, _ in pairs):
                stage = 'one_to_one_conflict'
            elif len(boxes) and np.any((ios[local] > MATCH_IOS_MIN) & tied):
                stage = 'ambiguous_mask_owner'
            elif near:
                stage = 'mask_overlap_below_threshold'
            else:
                stage = 'no_number_detection_near_player'
            records.append(dict(frame=frame, time_s=frame / manifest['source']['fps'],
                                track_id=int(row.track_id), player_id=int(row.player_id),
                                team=int(row.team), is_goalkeeper=bool(row.is_goalkeeper),
                                number=row.number, number_live=row.number_live,
                                stage=stage, number_row=number_row, read=text,
                                best_ios=float(ios[local].max()) if len(boxes) else 0.,
                                nearby_number_boxes=len(near), mask_row=int(row.mask_row)))
        if frame % 2500 == 0:
            print(f'audited frame {frame}', flush=True)
    assert validated_reads == set(read_keys), len(set(read_keys) - validated_reads)
    expected_attempts = {(int(f), int(n)) for f, hit in raw.items() for n in hit}
    assert counted_attempts == expected_attempts
    opportunities = pd.DataFrame(records)
    opportunities.to_parquet(output / 'opportunities.parquet', index=False)
    pd.DataFrame(box_records).to_parquet(output / 'number-boxes.parquet', index=False)
    episodes = []
    events = pd.read_parquet(bundle / 'events.parquet')
    for track_id, rows in tracks.groupby('track_id', sort=True):
        rows = rows.sort_values('frame')
        # Identity events are extra boundaries even when the tracker keeps its id.
        breaks = set(events.loc[(events.player_id == track_id) &
                                events.type.isin(['reid', 'suspected_id_switch']), 'frame'])
        changed = (rows.frame.diff().ne(1) | rows.number.isna().ne(rows.number.isna().shift()) |
                   rows.player_id.ne(rows.player_id.shift()) | rows.frame.isin(breaks))
        for _, span in rows.groupby(changed.cumsum()):
            if span.number.notna().any() or len(span) < 25:
                continue
            first, last = int(span.frame.min()), int(span.frame.max())
            obs = opportunities[(opportunities.track_id == track_id) &
                                opportunities.frame.between(first, last)]
            family_reads = reads[(reads.player_id == int(span.player_id.iloc[0])) &
                                 reads.frame.between(first, last)]
            episodes.append(dict(track_id=int(track_id), player_id=int(span.player_id.iloc[0]),
                                 first=first, last=last, frames=len(span),
                                 team=int(span.team.mode().iloc[0]),
                                 is_goalkeeper=bool(span.is_goalkeeper.mean() >= .5),
                                 stages=dict(Counter(obs.stage)),
                                 reads=dict(Counter(obs.read.dropna()[obs.read.dropna() != ''])),
                                 family_reads=dict(Counter(family_reads.value))))
    (output / 'missing-episodes.json').write_text(json.dumps(episodes, indent=2) + '\n')
    summary = dict(bundle=str(bundle.resolve()), source_commit=manifest['provenance']['git_commit'],
                   audit_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   input_sha256=before, observations=len(tracks),
                   missing_observations=int(tracks.number.isna().sum()),
                   stage_counts=dict(Counter(opportunities.stage)),
                   missing_stage_counts=dict(Counter(opportunities.loc[opportunities.number.isna(), 'stage'])),
                   accepted_reads_verified=len(validated_reads),
                   reader_attempts_verified=len(counted_attempts),
                   missing_episodes_at_least_one_second=len(episodes))
    after = {name: digest(bundle / name) for name in input_names}
    assert before == after, 'Input bundle changed during audit'
    summary['input_tables_unchanged'] = True
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.bundle, args.work, args.output)
    audit_labels(args.bundle, args.work, args.output)
