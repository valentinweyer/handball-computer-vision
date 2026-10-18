"""The same cached RF-DETR inputs and verified identities as SAM2 comparisons."""
from experiments.cameltrack.runtime import ROOT

CASES = {
    'felix': ('FelixClaar', 'team_comparison/.FelixClaar_detections_v1.npz',
              '.FelixClaar_ref_masks', 'FelixClaar_reference', ()),
    'han': ('Han-Ber4_cached', 'team_dataset/.Han-Ber4_detections_v1.npz',
            '.Han-Ber4_sam2_masks', 'HanBer4_sam2_reference', (12, 14)),
    'bhc': ('BHC-FAG_window_cached', 'team_dataset/.BHC-FAG_window_detections_v1.npz',
            '.BHC-FAG_window_ref_masks', 'BHC-FAG_window_reference', ()),
}


def inputs(case):
    stem, detections, reference, labels, excluded = CASES[case]
    return {'frames': ROOT / 'data/cache/frames' / stem,
            'video': ROOT / 'data/raw' / f'{stem}.mp4',
            'detections': ROOT / 'outputs' / detections,
            'reference': ROOT / 'source' / reference,
            'labels': ROOT / 'runs/tracklet_labels' / labels / 'answers.txt',
            'exclude_ids': excluded}
