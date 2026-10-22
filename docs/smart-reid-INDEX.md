# Smart re-ID: index of the experiment record

`prototype/smart-reid-shadow` is the permanent record of the smart re-ID work:
method, locks, measurements, labels and tests. It is **never merged as-is**.
Production integration would be a clean, opt-in reimplementation in `src/`.

This page maps each lock to its test, result and report. It also says where the
data lives and which files cannot be regenerated.

## Data split (fixed 2026-10-20)

- **Test**, used only on a locked candidate and never fed back into tuning:
  - Melsungen–Löwen, Magdeburg–Wetzlar and Stuttgart–Erlangen (the "higher"
    bank). Spent on 2026-10-21 by the coverage_v1 test.
  - The untouched reserve: DRE–MTM and Vinnhorst–Potsdam (both
    `*_66m00_10min`).
- **Development**, everything else:
  - the 8 split-v1 tuning matches;
  - Kiel–Lemgo, Eisenach–Hamburg, Gummersbach–Löwen;
  - Saarlouis, Bielefeld, Balingen II, Würzburg;
  - extra windows of these matches. Kiel 56:00 and 76:00 and Eisenach 57:00 and
    76:00 were processed 2026-10-21 as gallery-only footage.

## Locks, tests and results (chronological)

Tags of the form `smart-reid/*` mark each lock and each result commit. Lock
hashes are the first 12 hex digits of the sha256 of the file in
`experiments/teammate_reid/locked/`.

| lock (tag) | file, sha256 | tested on | result (tag) | report |
| --- | --- | --- | --- | --- |
| stage-1 thresholds (`smart-reid/stage1-lock`, e3a3bb3) | `thresholds.json` 09c437d89ebe | 2 held-out windows | **FAIL** on coverage (`smart-reid/stage1-fail`, 22f3051) | `smart-reid-shadow-2026-10-18.md` |
| full-crop stripe rule (`smart-reid/stripes-lock`, 9f0689f) | `thresholds_fresh_stripes.json` 4984d823eebf | fresh windows 3–4 | **FAIL**, precision short by 0.002 (`smart-reid/stripes-fail`, adde2ae) | `smart-reid-shadow-2026-10-18.md` |
| criterion v2 (23c976f) | `criterion_v2.json` 2d90cb3e19fa | defines pass (P ≥ 0.95, unknown FMR ≤ 3%, coverage ≥ 0.20) | n/a | n/a |
| absolute open-set rule (`smart-reid/openset-lock`, 11f776f) | `thresholds_sealed.json` 9adc84269cd0 | sealed bank | **FAIL** (`smart-reid/openset-fail`, 0c4ed0f) | `smart-reid-shadow-2026-10-18.md` (§ sealed test) |
| split v1 (`smart-reid/split-v1-lock`, fb36758) | `split_v1.json` 5c77fcdf6997 | 3 eval/ matches (Kiel, Eisenach, Gummersbach) | **PASS** (`smart-reid/split-v1-pass`, b1b81dd); Stage 2 on the same bank (264238c) | `smart-reid-shadow-2026-10-19-split.md` |
| split v2 (`smart-reid/split-v2-lock`, 06aabdb) | `split_v2.json` 84aae4f1bf94 | 4 new lower-league matches | **PASS** 113/113, unknown FMR 0/229, coverage 0.516 (`smart-reid/split-v2-pass`, 4697b7b) | `smart-reid-shadow-2026-10-20-final.md` |
| (split v2, Stage 2 on the final bank) | same | the same 4 matches, unresolved segments | 47/48 fully reviewed, blind 45/48 (`smart-reid/stage2-final-reviewed`, 5e47042) | `smart-reid-shadow-2026-10-20-final.md` |
| coverage_v1 (`smart-reid/coverage-v1-lock`, f3cce93) | `coverage_v1.json` c0e6e1beb443 | higher bank | **INCONCLUSIVE**: 3 added links, all correct, no added error (`smart-reid/coverage-v1-inconclusive`, 93c55a3) | `smart-reid-coverage-fresh-2026-10-21.md` |

Development reports, with no test spent:

- `teammate-reid-prototype-2026-10-18.md` (the original benchmark)
- `smart-reid-coverage-2026-10-20.md`
- `smart-reid-coverage2-2026-10-20.md`
- `smart-reid-extra-galleries-2026-10-21.md` (more of the match as gallery: real recall gain on all 7 dev matches; no size-robust variant is error-free, nothing lockable)
- `smart-reid-ceiling-2026-10-22.md` (63% of unnamed time belongs to players with a window gallery, against ~8–10% named today; the gate, not gallery availability, is the limit)

## Label provenance (kept apart everywhere)

- **User, blind.** Stage 2 labels:
  - `annotations/stage2_test_v1_human_v1.json`
  - `annotations/stage2_final_human_v1.json`
- **User, blind re-check:** `annotations/stage2_test_v1_human_v1_recheck.json`.
- **User, post-unsealing adjudication (not blind):**
  - `annotations/stage2_test_v1_human_v1_adjudication.json`
  - `annotations/stage2_final_adjudication_v1.json`
- **Assistant labels.** These are never human ground truth.
  - Labels:
    - `annotations/coverage_dev_assistant_labels_v{1,2}.json`
    - `annotations/coverage_fresh_assistant_labels_v1.json`, plus its team-id
      erratum
  - Gallery and contact-sheet reviews: `annotations/*_review_v1.json`
  - The agreement audit: `annotations/stage2_final_agreement_audit_v1.json`

## Irreplaceable files

These files cannot be regenerated: rerunning would either change them or break
the blind order. All are copied into
`experiments/teammate_reid/annotations/` (sealed decisions in `sealed/`). The
originals stay read-only under `runs/smart-reid-shadow/`.

| file | original path | sha256 (12) |
| --- | --- | --- |
| Stage 2 test_v1 sealed proposals | `stage2/proposals_SEALED.json` | see `annotations/sealed/` |
| Stage 2 final sealed proposals | `stage2_final/proposals_SEALED.json` | see `annotations/sealed/` |
| coverage fresh sealed decisions | `coverage_fresh/decisions_SEALED.json` | `.sha256` beside it |
| coverage fresh assistant labels | `coverage_fresh/assistant_labels_v1.json` | f011e349feea |
| coverage fresh team erratum | `coverage_fresh/assistant_labels_v1_erratum.json` | e7af6f901543 |
| split v1 and split v2 one-shot results | `split_v1/test.json`, `final/test.json` | n/a |

Each frozen label file has a `.sha256` file next to its original. The scorers
refuse to run on a file that has changed.

## Data on disk (gitignored; about 18 GB in `runs/smart-reid-shadow/`)

| Directory | Contents |
| --- | --- |
| `dev/`, `dev_b/`, `openset_11*/`, `symmetric/`, `calibration/` | Development mining, features, sheets and benchmark grids |
| `stage1/`, `fresh/`, `sealed/`, `sealed_preview*/`, `split_v1/`, `final/`, `higher/` | Per-bank mining (`fragments.json`, `crops.json`, `features*.npz`, `sheets/`, `review.json`) |
| `stage2/`, `stage2_final/`, `stage2_higher/` | Stage 2 populations, review clips and labelling pages |
| `coverage_dev/`, `coverage_dev2/`, `coverage_fresh/` | Evidence stores, rule grids, blind queues and labels |
| `logs/` | Every background run |
| `run_*.sh`, `mine_*.sh`, `stage2_final.sh`, `videos_final.sh` | Background drivers |

Bundles come from production `process_video` (main checkout d0fea4d). They live
in `runs/bundles/<window>/`, with resumable stages in `data/interim/<window>/`.
Do not delete these. Regenerating a 10-minute window costs about 2–2.5 GPU
hours.

## Reproduce

Python is the main venv. Never run `uv sync`. Tests:

```bash
PYTHONPATH=src:. $PY -m pytest -q tests/unit/test_teammate_reid.py
```

- Mining, features and sheets per bank:
  `python -m experiments.teammate_reid.run --set <bank> --out <dir> {mine,extract,sheets}`
  (`--only a,b`). Then `encoders.extract` for SportsReID.
- Split v2 one-shot: `final_test.py`. It refuses a second run.
- Stage 2:
  `stage2 --bank {test_v1,final,higher} {population,propose,review,serve,score}`.
- Coverage development: `coverage.py` and `coverage2.py` (`extract`,
  `evaluate`), with rules in `coverage_rules.py` and `coverage2_rules.py`.
- Coverage fresh test: `coverage2_fresh.py {decide,queue,score}`. Each step
  refuses to redo what is sealed.

## Next (development data only)

- Gallery-only extra windows (Kiel and Eisenach) to test "more of the match →
  more galleries".
- Fewer reads with cross-segment corroboration.
- Motion joining of segments before matching.
- A tail-switch guard. Both fresh-bank errors were short tail switches.
- Team reference crops in the fresh-test queue strips.

Rules for this work:

- No per-match tuning.
- No on-court player counting.
- No fine-tuning until a locked candidate shows it helps.
- A rule needs a larger test (for example, pooling the reserve) to get past
  INCONCLUSIVE when it adds few links.
