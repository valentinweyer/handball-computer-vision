"""Score a completed replay with the repository's identity evaluator (CPU only)."""
import argparse
import json
import os
from pathlib import Path


def evaluate(run, replay):
    from experiments.cameltrack.runtime import bootstrap
    from experiments.cameltrack.cases import inputs
    bootstrap()
    from scripts.evaluate_tracker_identity import (
        Sam2ReplayTracker, load_reference, parse_labels, run_tracker, score,
    )
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Only score completed runs; incomplete prefixes must not look like full results')
    paths = inputs(manifest['case'])
    valid = parse_labels(paths['labels'])
    keep = set(valid) - set(paths['exclude_ids'])
    reference = load_reference(paths['reference'], 0, valid)
    reference = {frame: truth for frame, truth in reference.items()
                 if 0 <= frame < manifest['processed_frames']}
    denominator = sum(sum(identity in keep for identity in truth) for truth in reference.values())
    if not denominator:
        raise ValueError('No trusted player observations in evaluated prefix')
    matches = run_tracker(lambda: Sam2ReplayTracker(replay), reference,
                          paths['video'], paths['detections'], 'cpu')
    result = score(matches, keep)
    matched = result['matched_detections']
    assert matched <= denominator
    result.update(reference_player_observations=denominator,
                  coverage_pct=100 * matched / denominator,
                  correct_identity_pct=100 * (matched-result['wrong_identity_frames']) / denominator,
                  missing_player_observations=denominator-matched,
                  evaluated_frames=manifest['processed_frames'], case=manifest['case'],
                  metric='Existing IoU>=0.5 greedy box matching, dominant identity per track; human-verified reference spans')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--baseline', type=Path, help='optional SAM2 run directory containing replay.npz and manifest.json')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    result = {'cameltrack': evaluate(args.run, args.run / 'replay.npz')}
    if args.baseline:
        from experiments.cameltrack.runtime import sha256
        candidate = json.loads((args.run / 'manifest.json').read_text())
        baseline = json.loads((args.baseline / 'manifest.json').read_text())
        for key in ['case', 'video_sha256', 'detections_sha256']:
            if baseline[key] != candidate[key]:
                raise ValueError(f'Baseline has different {key}')
        if baseline['jpeg_count'] < candidate['processed_frames']:
            raise ValueError('Baseline does not span the candidate prefix')
        result['sam2_same_prefix'] = evaluate(args.run, args.baseline / 'replay.npz')
        result['baseline_manifest_sha256'] = sha256(args.baseline / 'manifest.json')
    (args.run / 'identity_score.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
