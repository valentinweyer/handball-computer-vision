# Smart re-ID: fresh test of coverage_v1 on the higher-league bank (2026-10-21)

Branch `prototype/smart-reid-shadow`, never merged. This is the one-shot test of
`experiments/teammate_reid/locked/coverage_v1.json`:

- **Lock commit:** f3cce93, sha256 `c0e6e1be…`.
- **Driver commit:** 4ee3468.
- **Bank:** Melsungen–Löwen, Magdeburg–Wetzlar, Stuttgart–Erlangen, each the
  66:00 window, 10 min. None of the three had been inspected before the lock.
- **Reserve:** DRE–MTM and Vinnhorst–Potsdam stay untouched.

## Verdict: INCONCLUSIVE

The lock prescribes INCONCLUSIVE when fewer than 5 added links are verified
correct.

- coverage_v1 added **3 names**, and all 3 are verified correct.
- It added no error, lost no link, and changed no name.
- It made no synthetic-unknown false match (0/37, the same as v2).

The coverage target is met on paper: verified-correct names rise **+15%**
(20 → 23), and the target is ≥ 10%. On 3 links that is weak evidence.

With n = 3 and zero errors, the one-sided 95% lower bound on added-link
precision is only **0.37**. This shows no gross failure. It does not show that
the change is safe.

## What happened, paired per segment

Population (Stage 2 definition): 368 segments. 363 of them have no named row in
the baseline. In total, 4,412 player-seconds are unnamed.

| | v2 (split_v2) | coverage_v1 |
| --- | ---: | ---: |
| names given | 38 | 41 |
| correct (assistant-verified) | 20 | 23 |
| wrong known player / unknown named | 0 / 0 | 0 / 0 |
| mixed track (box switches person) | 2 | 2 |
| unreadable (unverified) | 16 | 16 |
| precision, correct / (correct + errors) | 0.909, CI 0.73–1.00 | 0.920, CI 0.77–1.00 |
| segment coverage | 5.4% | 6.3% |
| newly correct player-time | 170 s (3.8%) | 205 s (4.6%) |
| synthetic unknowns falsely matched | 0 / 37 | 0 / 37 |

- **Added links** (all three are lone-candidate cases; none came from the
  duplicate-track exemption):
  - Stuttgart S279: #6
  - Stuttgart S266: #7
  - Melsungen S192: #8
- **Both rules name 38 segments identically.**
- **The two mixed errors are shared by both rules,** so the candidate did not
  cause them:
  - Melsungen S176 (H002) is #8 for 19.8 s, then the box slides onto a falling
    Löwen player for about 0.6 s.
  - Melsungen S177 (H049) is #14 for about 8 s, then the box ends on Löwen #80.

  Both are short tail switches, which the half-split guard does not catch. Under
  a majority-identity reading they would count as correct.
- **Per match** (correct, then unreadable, for v2 → coverage_v1):

  | Match | Segments | Correct | Unreadable |
  | --- | ---: | --- | ---: |
  | Stuttgart | 172 | 14 → 16 | 7 |
  | Melsungen | 82 | 4 → 5 (+2 mixed) | 2 |
  | Magdeburg | 114 | 2 → 2 | 7 |

  In Magdeburg most names cannot be verified: green backs at broadcast distance.

## Why so few links

This bank is much harder for the matcher than the lower-league one. v2 names
10% of the population here, against 14% on the spent banks.

- **Almost everything is zero-known.** 363 of 368 segments have no number at
  all in the baseline, so there are few vouched galleries to match against.
- **Stuttgart was processed with the court pause,** as the lock specifies
  (production `process_video`). Court was in view only 66% of the time, so less
  footage became tracks. The no-pause run exists but was not used.

Expected added links were 5–15. The bank delivered 3. Waiting for a bigger fresh
bank, or pooling with the reserve, is the only way this rule gets a
non-inconclusive verdict.

## Procedure as run (all hashes in `annotations/coverage_fresh_result_v1.json`)

1. Production `process_video` on the main checkout at d0fea4d. Then mine,
   extract features and build sheets (`mine_higher.sh`).
2. **Gallery sheet review before any decision.** All 47 identities and 198
   fragments, with suspect fragments re-inspected at full crop size. Nothing was
   excluded. Notes are in `annotations/higher_gallery_review_v1.json`, including
   a look-alike pair in Stuttgart, blue #19 and #22.
3. Population (`stage2 --bank higher population`) and evidence store
   (`coverage2 extract --banks higher`).
4. v2 and coverage_v1 decisions sealed together in `decisions_SEALED.json`.
5. Blind queue: the 41 segments either rule named, plus 40 seeded random
   segments, under shuffled codes H000–H080.
6. **Assistant labels**: 81 items, frozen as
   `annotations/coverage_fresh_assistant_labels_v1.json` (sha256
   `f011e349…`) before any decision was opened. These are assistant labels,
   **not human ground truth**.
   - 37 have a readable number.
   - 36 are unreadable.
   - 6 are mixed.
   - 2 are not a player (the referee, and a box on the advertising board).
7. Scored once (`coverage2_fresh score`).

### Disclosed correction: team ids in Magdeburg–Wetzlar

Team ids are anonymous clusters per video, and the blind strips showed no team
reference. For Magdeburg–Wetzlar I wrote white Wetzlar players as team 1, except
one (H013), which I wrote as 0. The labels were inconsistent with each other.

The gallery fixes the mapping without looking at any decision:

| Match | Cluster 0 (torso mean BGR) | Cluster 1 (torso mean BGR) |
| --- | --- | --- |
| Magdeburg–Wetzlar | (151, 115, 109) = white | (70, 78, 60) = green |
| Melsungen | red | dark |
| Stuttgart | white | blue |

The Melsungen and Stuttgart labels already matched.

I found this after freezing, when I opened the decisions to choose items for your
adjudication. The frozen label file is unchanged. A separate frozen erratum
remaps **only the team field** of 8 white-Wetzlar items to 0:
`annotations/coverage_fresh_assistant_labels_v1_erratum.json` (sha256
`e7af6f90…`). The scorer applies it and records both hashes.

Without the erratum, two correct v2 names (Wetzlar #23, H066 and H080) would
have scored as wrong. No added link is in Magdeburg, so the verdict does not
depend on the erratum.

**Tooling fix for next time:** queue strips need team reference crops, as the
Stage 2 pages had.

## Open for you (optional, reported apart)

The verdict cannot change: there are only 3 added links, and all are verified.
Your look would firm up the shared precision figures.

- **Mixed or not?** H002 (#8, 0.6 s tail) and H049 (#14, tail onto Löwen #80).
- **Unreadable, but named by both rules:**

  | Item | Name given | Note |
  | --- | --- | --- |
  | H001 | Löwen #21 | |
  | H003 | #10 | |
  | H013 | Wetzlar #19 | |
  | H024 | Melsungen #33 | |
  | H033 | Erlangen #44 | |
  | H034 | Magdeburg #11 | |
  | H041 | Magdeburg #11 | |
  | H048 | Stuttgart #22 | |
  | H053 | #6 | |
  | H064 | Wetzlar #23 | I read 23 or 11 |
  | H067 | Wetzlar #13 | |
  | H070 | #10 | |
  | H071 | Wetzlar #19 | |
  | H073 | #6 | |
  | H074 | #6 | |
  | H078 | Wetzlar #19 | I read 19 or 14 |

Each of these is a name only in the sense that it cannot be verified. None is a
known error.

## Reproduce

```bash
PYTHONPATH=src:. $PY -m experiments.teammate_reid.stage2 --bank higher population
PYTHONPATH=src:. $PY -m experiments.teammate_reid.coverage2 extract --banks higher
PYTHONPATH=src:. $PY -m experiments.teammate_reid.coverage2_fresh decide   # refuses if sealed
PYTHONPATH=src:. $PY -m experiments.teammate_reid.coverage2_fresh queue
PYTHONPATH=src:. $PY -m experiments.teammate_reid.coverage2_fresh score    # refuses if result.json exists
```

Outputs are in `runs/smart-reid-shadow/coverage_fresh/`: sealed decisions, queue,
pages, frozen labels and erratum (all read-only), and `result.json` with
per-segment outcomes.
