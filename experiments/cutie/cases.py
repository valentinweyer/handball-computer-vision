"""Existing validation videos, RF-DETR caches and trusted identity references."""
from experiments.cutie.adapter import ROOT

CASES = {
    'felix': ('FelixClaar', 'team_comparison/.FelixClaar_detections_v1.npz',
              'team_comparison/.FelixClaar_team.pkl', '.FelixClaar_ref_masks',
              'FelixClaar_reference', ()),
    'han': ('Han-Ber4_cached', 'team_dataset/.Han-Ber4_detections_v1.npz',
            'team_correction_mcbyte/.Han-Ber4_team.pkl', '.Han-Ber4_sam2_masks',
            'HanBer4_sam2_reference', (12, 14)),
    'bhc': ('BHC-FAG_window_cached', 'team_dataset/.BHC-FAG_window_detections_v1.npz',
            'team_confidence_v2/.BHC-FAG_team.pkl', '.BHC-FAG_window_ref_masks',
            'BHC-FAG_window_reference', ()),
}


def inputs(case):
    stem, detections, team, reference, labels, excluded = CASES[case]
    return {'frames': ROOT/'data/cache/frames'/stem,
            'video': ROOT/'data/raw'/f'{stem}.mp4',
            'detections': ROOT/'outputs'/detections,
            'team': ROOT/'outputs'/team,
            'reference': ROOT/'source'/reference,
            'labels': ROOT/'runs/tracklet_labels'/labels/'answers.txt',
            'exclude_ids': excluded}
