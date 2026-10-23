# Tracker foundation event review

Compare unchanged ARC OSNet localconf, existing colour-gated SORT (gate .28,
activation .5, buffer30), and stored production SAM2 on all twelve previously
reviewed Balingen–Kiel overlap windows. No tuning or repair rules.

ARC and SORT consume the same 15000-frame stride-one expanded-v1 cache at .30.
SAM2 is the stored production pipeline (commit d0fea4d, detector stride5,
check_every10, lifecycle and PRTReID), not a pure association-only ablation.
Use tracks.parquet track_id before jersey-number folding; never player_id.

SORT runs from frame zero to preserve history. ARC/SAM2 replay full-run outputs.
Render three separate 1080p videos per event. Panel labels rotate deterministically.
The key is outside the review folder. Anchor X/Y is chosen from pre-contact ARC
geometry, then each method is matched once (IoU >= .5, margin >= .1); ambiguous
matches remain unassigned. No remapping after the overlap. Human reviewer must
verify target identity; anchor matching is not ground truth.

Per-panel verdict: correct / error / uncertain / out of scope. Failure categories
are human hypotheses, not automatic diagnoses. Temporary recovered swaps count.
Exclude bench/off-court people. Export JSON; retain earlier ARC judgments.

Stop after delivering a verified package. Do not tune, repair, or switch defaults.
This is one ARC-selected dev sample with detector training exposure; no overall
accuracy/HOTA/IDF1 claims. Use existing three-clip reference results as separate
supporting evidence, without pooling incompatible denominators.

Run with repository src and root on PYTHONPATH, existing pipeline venv, CPU only:
python experiments/tracker_foundation_review/run.py --root RUN_DIRECTORY
Pre-create protocol.json and copy review-template.html into RUN_DIRECTORY.
All input SHA256 values checked before and after; every output video fully decoded.
