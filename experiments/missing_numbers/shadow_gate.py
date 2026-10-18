"""Compare predeclared small-number gates without changing production inputs.

The human-labelled native crop set measures read quality across six clips.
The recorded Kiel tracks/links are held fixed for a separate temporal replay;
this cannot predict tracking or linking decisions in a fresh extraction.
"""
from __future__ import annotations

import copy
import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pandas as pd

from handball_cv.jersey.identity import is_valid_number
from handball_cv.jersey.parseq_backend import load_jersey_parseq, read_crops
from handball_cv.pipeline.result import sha256, load
from handball_cv.pipeline.tables import tracks_table
from handball_cv.teams.model import crop_quality
from handball_cv.tracking.geometry_cache import GeometryCache
from scripts.render_result import court_fits

ROOT = Path('runs/number-crop-shadow-2026-10-18')
AUDIT = Path('runs/missing-numbers-audit-2026-10-18')
WORK = Path('data/interim/Kiel_Lemgo_66m00_10min_expanded_validation_v4')
BUNDLE = Path('runs/bundles/Kiel_Lemgo_66m00_10min_expanded_validation_v4_backfill_fix')
DATASET = Path('runs/number_eval_1080p/dataset.json')
LABELS = Path('data/annotations/jersey/number_eval_1080p_labels.json')
CHALLENGE = Path('experiments/missing_numbers/challenge_labels.json')
ANNOTATION_REVIEW = Path('experiments/missing_numbers/annotation_review.json')
POSTHOC = Path('experiments/missing_numbers/posthoc_labels.json')
CHECKPOINT = Path('models/jersey_parseq/parseq_original.ckpt')


@dataclass(frozen=True)
class Gate:
    name: str
    minimum_area: int
    small_confidence: float = .5


# Fixed before comparing benchmark outputs. Larger, currently accepted crops
# always retain their original 0.5 threshold, including the guarded candidate.
GATES = [Gate('baseline400', 400), Gate('area350', 350), Gate('area300', 300),
         Gate('area250', 250), Gate('area200', 200), Gate('guarded250', 250, .95)]


def accepts(row, gate):
    area = int(row['width']) * int(row['height'])
    return (min(row['width'], row['height']) >= 10 and area >= gate.minimum_area
            and row['contrast'] >= 12 and row['sharpness'] >= 20
            and row['confidence'] >= (gate.small_confidence if area < 400 else .5)
            and is_valid_number(str(row['text']).strip()))


def human_metrics(rows, gate, baseline=None):
    accepted = rows.apply(lambda r: accepts(r, gate), axis=1)
    readable = rows.status.eq('readable') & ~rows.not_number
    negative = rows.not_number
    unknown = rows.status.eq('unreadable') & ~negative
    correct = readable & accepted & rows.text.eq(rows.truth)
    wrong = readable & accepted & ~rows.text.eq(rows.truth)
    report = dict(crops=len(rows), readable=int(readable.sum()),
                  correct=int(correct.sum()), wrong=int(wrong.sum()),
                  abstained_readable=int((readable & ~accepted).sum()),
                  not_number=int(negative.sum()), false_accepts=int((negative & accepted).sum()),
                  unreadable_number_outputs=int((unknown & accepted).sum()))
    if baseline is not None:
        extra = accepted & ~baseline
        report.update(extra_correct=int((extra & correct).sum()),
                      extra_wrong=int((extra & wrong).sum()),
                      extra_false_accepts=int((extra & negative).sum()),
                      extra_unverified=int((extra & ~readable & ~negative).sum()))
    return report


def benchmark_human():
    data = json.loads(DATASET.read_text())
    labels = json.loads(LABELS.read_text())['labels']
    images, records = [], []
    for sample in data['samples']:
        path = DATASET.parent / sample['crop_path']
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(rgb)
        q = crop_quality(rgb)
        label = labels[str(sample['index'])]
        records.append(dict(index=sample['index'], clip=sample['source_clip'],
                            status=label['status'], truth=label.get('value'),
                            not_number=label.get('box_status') == 'not_number',
                            production_quality_accepted=q.accepted,
                            width=q.width, height=q.height, contrast=q.contrast,
                            sharpness=q.sharpness, crop_path=str(path)))
    model, transform = load_jersey_parseq(CHECKPOINT)
    raw = read_crops(model, transform, images)
    del model
    for row, (text, confidence) in zip(records, raw):
        row.update(text=str(text).strip(), confidence=float(confidence))
    rows = pd.DataFrame(records)
    rows.to_parquet(ROOT / 'human-crops.parquet', index=False)
    baseline = rows.apply(lambda r: accepts(r, GATES[0]), axis=1)
    expected = rows.production_quality_accepted & rows.confidence.ge(.5) & rows.text.map(is_valid_number)
    assert baseline.equals(expected), 'Baseline differs from the production predicate'
    reports = {}
    for gate in GATES:
        admitted = rows.apply(lambda r: accepts(r, gate), axis=1)
        assert not (baseline & ~admitted).any(), 'Candidate removes a currently accepted crop'
        overall = human_metrics(rows, gate, baseline)
        reports[gate.name] = dict(overall=overall,
                                  by_clip={clip: human_metrics(group, gate,
                                                              baseline.loc[group.index])
                                           for clip, group in rows.groupby('clip')})
        print('HUMAN', gate.name, json.dumps(overall), flush=True)
    (ROOT / 'human-results.json').write_text(json.dumps(reports, indent=2) + '\n')
    sensitivity(rows)
    return reports


def sensitivity(rows):
    """Exclude disputed annotations separately; never rewrite primary scores."""
    excluded = json.loads(ANNOTATION_REVIEW.read_text())['excluded_indices']
    remaining = rows[~rows['index'].isin(excluded)]
    baseline = remaining.apply(lambda r: accepts(r, GATES[0]), axis=1)
    results = {g.name: human_metrics(remaining, g, baseline) for g in GATES}
    (ROOT / 'human-sensitivity.json').write_text(json.dumps(
        dict(excluded_indices=excluded, results=results), indent=2) + '\n')
    return results


def injected_reads(run, accepted, boxes):
    """Resolve each NEW read's owner using only links made before that frame."""
    aliases = {}
    links = sorted((e for e in run['identity_events'] if e['type'] == 'link'),
                   key=lambda e: e['frame'])
    position = 0

    def canonical(value):
        seen = set()
        while value in aliases and value not in seen:
            seen.add(value)
            value = aliases[value]
        return value

    out = []
    for row in accepted.sort_values(['frame', 'number_row']).itertuples():
        while position < len(links) and links[position]['frame'] < row.frame:
            event = links[position]
            source, target = canonical(event['player_id']), canonical(event['into_player_id'])
            if source != target:
                aliases[source] = target
            position += 1
        box = boxes.loc[(row.frame, row.number_row), ['x1', 'y1', 'x2', 'y2']]
        out.append(dict(frame=int(row.frame), track_id=int(row.track_id),
                        player_id=int(canonical(int(row.track_id))), value=str(row.text),
                        confidence=float(row.confidence), number_row=int(row.number_row),
                        box=box.to_list()))
    return out


def challenge_metrics(tracks, challenge, baseline):
    reports = []
    for case in challenge['sequences']:
        rows = tracks[(tracks.track_id == case['track_id']) &
                      tracks.frame.between(case['first'], case['last']) &
                      ((tracks.x2 - tracks.x1) >= 20) & ((tracks.y2 - tracks.y1) >= 70)]
        ref = baseline.loc[rows.index, 'number']
        values = rows.number
        known = values.notna()
        good = values.eq(case['number'])
        bad = known & ~good
        ref_good = ref.eq(case['number'])
        ref_bad = ref.notna() & ~ref_good
        reports.append(dict(**case, observations=len(rows), correct=int(good.sum()),
                            wrong=int(bad.sum()), missing=int((~known).sum()),
                            newly_correct=int((good & ~ref_good).sum()),
                            newly_wrong=int((bad & ~ref_bad).sum()),
                            lost_correct=int((ref_good & ~good).sum()),
                            first_wrong_frame=int(rows.loc[bad, 'frame'].min()) if bad.any() else None,
                            numbers=values.dropna().value_counts().to_dict()))
    keys = ['observations', 'correct', 'wrong', 'missing', 'newly_correct', 'newly_wrong', 'lost_correct']
    return dict(overall={key: sum(row[key] for row in reports) for key in keys}, sequences=reports)


def replay():
    probe = pd.read_parquet(AUDIT / 'crop-gate-probe.parquet').rename(
        columns={'probe_text': 'text', 'probe_confidence': 'confidence'})
    boxes = pd.read_parquet(AUDIT / 'number-boxes.parquet').set_index(['frame', 'number_row'])
    run = json.loads((WORK / 'track.json').read_text())
    # Check the owner mapping against observed runtime behavior before using
    # it for counterfactual reads (including reads made on link frames).
    originals = pd.DataFrame(run['number_reads']).rename(columns={'value': 'text'})
    owners = {(r['frame'], r['track_id'], r['number_row']): r['player_id']
              for r in run['number_reads']}
    replayed = injected_reads(run, originals, boxes)
    assert len(replayed) == len(owners)
    assert all(r['player_id'] == owners[(r['frame'], r['track_id'], r['number_row'])]
               for r in replayed)
    challenge = json.loads(CHALLENGE.read_text())
    posthoc = json.loads(POSTHOC.read_text())
    result = load(BUNDLE)
    old = result.tracks
    geometry = GeometryCache.load(BUNDLE / 'masks.npz')
    fits = court_fits(result.court)
    with np.load(WORK / 'track_extras.npz') as extras:
        team_confidence = extras['team_confidence']
    reports = {}
    posthoc_reports = {}
    for gate in GATES:
        mask = probe.apply(lambda row: accepts(row, gate), axis=1)
        additions = injected_reads(run, probe[mask], boxes)
        assert not pd.DataFrame(additions).duplicated(['frame', 'track_id']).any() if additions else True
        if not additions:
            assert gate == GATES[0]
            tracks = old
        else:
            candidate = copy.deepcopy(run)
            candidate['number_reads'] = sorted(run['number_reads'] + additions,
                                              key=lambda r: (r['frame'], r['number_row']))
            keys = [(r['frame'], r['track_id']) for r in candidate['number_reads']]
            assert len(keys) == len(set(keys))
            tracks = tracks_table(candidate, geometry, fits, result.fps, 5, team_confidence)
            pd.testing.assert_frame_equal(old.drop(columns=['number', 'number_live']),
                                          tracks.drop(columns=['number', 'number_live']))
        scored = challenge_metrics(tracks, challenge, old)
        field = tracks[~tracks.is_goalkeeper & tracks.number.notna()]
        assert not field.duplicated(['frame', 'team', 'number']).any()
        scored.update(extra_reads=len(additions),
                      full_run_labelled=int(tracks.number.notna().sum()),
                      full_run_removed=int((old.number.notna() & tracks.number.isna()).sum()),
                      full_run_changed=int((old.number.fillna('') != tracks.number.fillna('')).sum()))
        reports[gate.name] = scored
        posthoc_reports[gate.name] = challenge_metrics(tracks, posthoc, old)
        tracks[['frame', 'track_id', 'player_id', 'number', 'number_live']].to_parquet(
            ROOT / f'shadow-{gate.name}.parquet', index=False)
        (ROOT / 'shadow-results.json').write_text(json.dumps(reports, indent=2) + '\n')
        (ROOT / 'posthoc-results.json').write_text(json.dumps(posthoc_reports, indent=2) + '\n')
        print('SHADOW', gate.name, json.dumps({k: v for k, v in scored.items() if k != 'sequences'}), flush=True)
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--output', type=Path, default=ROOT,
                        help='new output directory; existing result files are refused')
    ROOT = parser.parse_args().output
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / 'human-results.json').exists()
    protected = [LABELS, DATASET, CHALLENGE, ANNOTATION_REVIEW, POSTHOC,
                 WORK / 'track.json', WORK / 'track_reads.json',
                 BUNDLE / 'manifest.json', BUNDLE / 'tracks.parquet']
    before = {str(path): sha256(path) for path in protected}
    provenance = dict(base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      checkpoint_sha256=sha256(CHECKPOINT), input_sha256=before,
                      gates=[asdict(g) for g in GATES],
                      scope='Human crop benchmark plus frozen-identity replay; no new tracking/link decisions',
                      benchmark_geometry='Stored native JPEG crops; exact production pixels used in Kiel probe')
    (ROOT / 'protocol.json').write_text(json.dumps(provenance, indent=2) + '\n')
    benchmark_human()
    replay()
    assert before == {str(path): sha256(path) for path in protected}
    provenance['protected_inputs_unchanged'] = True
    (ROOT / 'protocol.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Completed shadow comparison; protected inputs unchanged', flush=True)
