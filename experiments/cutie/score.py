"""Score completed Cutie/SAM2/CAMEL replays on exactly the same trusted frames."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutie', type=Path, required=True)
    parser.add_argument('--sam2', type=Path, required=True)
    parser.add_argument('--camel', type=Path)
    parser.add_argument('--output', type=Path, help='optional new score path; default inside the Cutie run')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from experiments.cutie.adapter import ROOT, bootstrap, sha256
    from experiments.cutie.cases import inputs
    bootstrap()
    from scripts.evaluate_tracker_identity import (
        Sam2ReplayTracker, load_reference, parse_labels, run_tracker, score,
    )
    runs = {'cutie': args.cutie, 'sam2': args.sam2}
    if args.camel:
        runs['cameltrack'] = args.camel
    manifests = {name: json.loads((run/'manifest.json').read_text()) for name, run in runs.items()}
    first = manifests['cutie']
    for name, manifest in manifests.items():
        assert manifest['status'] == 'complete', f'{name} run incomplete'
        for key in ['case', 'processed_frames', 'video_sha256', 'detections_sha256']:
            assert manifest[key] == first[key], f'{name}: different {key}'
    paths = inputs(first['case'])
    labels = paths['labels']
    valid = parse_labels(labels)
    keep = set(valid) - set(paths['exclude_ids'])
    reference = load_reference(paths['reference'], 0, valid)
    reference = {fid: truth for fid, truth in reference.items() if 0 <= fid < first['processed_frames']}
    denominator = sum(sum(identity in keep for identity in truth) for truth in reference.values())
    assert denominator > 0
    report = {'case': first['case'], 'excluded_reference_ids': list(paths['exclude_ids']), 'evaluated_frames': first['processed_frames'], 'reference_player_observations': denominator,
              'labels_sha256': sha256(labels),
              'metric': 'Existing IoU>=0.5 greedy box matching; dominant identity per track; missing observations count against correct identity',
              'reference_caveat': ('Han reference was generated with SAM2 and human-verified; it is independent of Cutie propagation but may favour SAM2' if first['case']=='han' else 'Reference was generated with SAM+Cutie and human-verified over trusted spans; it may favour Cutie'),
              'results': {}}
    for name, run in runs.items():
        matches = run_tracker(lambda: Sam2ReplayTracker(run/'replay.npz'), reference,
                              paths['video'], paths['detections'], 'cpu')
        result = score(matches, keep)
        matched = result['matched_detections']
        assert matched <= denominator
        result.update(coverage_pct=100*matched/denominator,
                      correct_identity_pct=100*(matched-result['wrong_identity_frames'])/denominator,
                      missing_player_observations=denominator-matched,
                      run=str(run.resolve()), manifest_sha256=sha256(run/'manifest.json'))
        report['results'][name] = result
    import torch
    assert not torch.cuda.is_initialized()
    output = args.output or args.cutie/'identity_score.json'
    if args.output and output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
