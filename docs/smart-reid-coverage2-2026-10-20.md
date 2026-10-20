# Smart re-ID coverage, round 2: candidate, lock and fresh higher-league test (2026-10-20/21)

Branch `prototype/smart-reid-shadow`, shadow only: no production code, defaults or
bundles are changed. Round 1 is `smart-reid-coverage-2026-10-20.md`. The candidate
lock is `experiments/teammate_reid/locked/coverage_v1.json`, sha256 `c0e6e1be…`,
commit f3cce93. It was written before any fresh-bank mining.

## Footage audit (before choosing anything)

| footage | status |
| --- | --- |
| Kiel, Eisenach, Gummersbach, Ludwigshafen, RNL, BAL-THW, BER-BHC, TBV-SCM, FAG-LEI, Altenholz, Pforzheim, and the four lower-league windows | used for tuning, testing or labelling: **development** |
| Melsungen-Loewen, Magdeburg-Wetzlar, Stuttgart-Erlangen (HBF, added 2026-10-20, 1080p50) | never cut, processed or viewed before this round: **fresh test bank `higher`** |
| DRE-MTM and Vinnhorst-Potsdam windows (cut 2026-10-19, never processed) | **sealed reserve** for the next independent test |

Before any processing, I checked a 6-thumbnail live-play strip per fresh window to
confirm second-half play, not ads (`runs/smart-reid-shadow/higher_check/`). The
fresh windows' input hashes were recorded before processing.

Clubs Melsungen, Loewen, Magdeburg and Wetzlar also appear in development data.
Galleries are built per match and the encoder is frozen, so no player identity is
learned across matches.

## Population and reference

The population is all 971 baseline-unresolved segments of the two spent banks,
under the Stage 2 definition (fewer than half the rows show a number, at least 3
clean crops, no goalkeepers).

- **Zero-known stratum:** 967 of the 971 segments, 14,793 of 14,897 s, have no
  baseline-named row at all. The "mostly unnamed" population is almost entirely
  zero-known, so the two strata give practically the same numbers.
- **Newly correct player-time:** counted as segment duration × the share of rows
  the baseline left unnamed.

The reference sources are kept apart and never merged:

- **The user's blind Stage 2 labels.**
- **The user's post-unsealing adjudication.** This is not blind.
- **My blind full-resolution labels, v1 and v2.** v1 has 137 segments plus 16
  controls. v2 (sha256 `68832165…`) has 20 segments plus 15 controls, including
  the regression cases. On the controls, every readable v2 label matched the
  user (10 of 10). These labels are development evidence, not human validation.
- **Disputed reads.** When my two reads of a segment disagree, it counts as
  *disputed* (unverified). It is never resolved by matching a prediction.

## Why true known players abstain

This uses the 130 human-labelled segments whose true player has a gallery. Under
the narrow rule, 39 are still not named:

| reason | n |
| --- | ---: |
| below the gate; the true player is the top candidate in 30 of them | 33 |
| lone candidate, score < 0.80 | 3 |
| truth vetoed (co-presence or number) | 4 |
| no candidate / halves disagree | 3 |

Their true-player scores were 0.62–0.80, below τ = 0.784. Nearly every abstention
is a correct top guess that is not confident enough.

Lowering the gate reaches them, but it also names confident look-alikes. Among
labelled segments where the matcher's top guess was a player with no gallery
(n = 11), raw scores reach 0.77 with margins of 0.10–0.11. Score and margin do not
separate those unknowns from correct names.

## Ablations (identical population and candidates; all scores re-run from one evidence store)

The store reproduces the sealed v2 decisions of the final bank exactly
(0 of 663 differ) and agrees with the round-1 cache on test_v1.

Column key: **wrong** = wrong known player; **unknown** = a player with no
gallery was named; **mixed** = mixed-person track; **unread** = unreadable or
disputed.

| rule | correct | wrong | unknown | mixed | unread | newly correct player-time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **v2 (locked)** | 100 | 1 | 2 | 2 | 33 | 1,170 s |
| lone candidate ≥ 0.80 only | 114 | 1 | 2 | 2 | 36 | 1,371 s |
| duplicate exemption only (IoU ≥ 0.5 on every frame) | 104 | 1 | 2 | 2 | 34 | — |
| duplicate exemption only (sustained duplicate track) | 104 | 1 | 2 | 2 | 34 | — |
| round-1 narrow (lone + IoU ≥ 0.5) | 119 | 1 | 2 | 2 | 37 | 1,446 s |
| **coverage_v1 (lone + sustained duplicate track)** | **119** | **1** | **2** | **2** | **37** | **1,446 s** |
| + tier 2 (raw ≥ 0.72–0.74 confirmed with digits hidden, crowd guard) | 123–133 | 1–2 | 2–3 | 3–4 | 47–56 | 1,514–1,614 s |
| + query outlier cleaning | 124 | 1 | 2 | 4 | 47 | — |
| + gallery outlier cleaning | 118 | 1 | 2 | 2 | 36 | — |
| all crops instead of 8 spread | 116 | 1 | 2 | 3 | 41 | — |
| strict-mask embedding only | 27–32 | 0 | 0 | 1 | 10 | — |

**Paired against v2, coverage_v1 adds 19 verified-correct names and 4 unreadable
ones, with no added error and no lost name.** In this data the two changes act
almost additively: 14 names come from the lone candidate, 4 from duplicate
tracks, and 1 from both.

**Duplicate test.**

- **Different players are never exempted.** Two kinds of negatives were checked:
  - 126 time-overlapping vouched fragments of different players;
  - 1,100 labelled co-presence events where the other box was a different
    player.

  None passes the sustained duplicate-track rule, and none passes the
  IoU ≥ 0.5-everywhere rule. 61 of those other-player events do reach
  IoU ≥ 0.5 on some frames. That is why a single-frame IoU test is not enough.
- **Some same-player cases are exempted:** 6 of 10 labelled same-player
  co-presence events pass.
- **The stricter rule is kept.** IoU ≥ 0.5 on every frame happens to give the
  same development result, but the lock uses the sustained rule because it has a
  defined negative test.

**Stress and digit checks.**

- **Synthetic unknowns** (the true player's gallery removed): 5 of 130 false
  matches under v2 and 5 of 130 under coverage_v1, so no increase. This is the
  case where the lone-candidate change could have hurt.
- **Strict mask:** every correct name, old and added, keeps the same winner with
  digits and a torso-and-shorts band hidden on both sides.
- **Digit boxes:** all 19 added names come from segments where the number
  detector found a box on some crop, but the number was never read. Raw crops
  can contain digits, so **this is not evidence of number-independent
  recognition.**

**Rejected.**

- **Tier 2:** under every setting it adds at least one mixed-person track, Kiel
  S105 (assistant-verified). The crowding guard removes most but not all.
- **Cleaning variants:** they add mixed names or lose correct ones.
- **Two-pass propagation:** not revisited. No purity safeguard was demonstrated,
  and tier 2 already shows that the crop-level purity checks (halves, thirds,
  support, two-cluster split, crowding) miss switches inside a track.

## Regression cases

| case | status |
| --- | --- |
| Balingen II S158 | true #7 (user), v2 and coverage_v1 assign #10. Still wrong. |
| Kiel S259 | mixed #4/#23 in both assistant reads; both rules name #4. Still a mixed-track error. |
| Eisenach S000 | both rules name #99. Assistant reads disagree (#55; then mixed #55/#99). Disputed. |
| Eisenach S011 | both rules name #99. Assistant reads disagree (#55; then #99). Disputed. |
| Bielefeld S241 | user #7. Assistant reads disagree (#6; then mixed #5/#8). Neither v2 nor coverage_v1 names it. Disputed; not resolved by any prediction. |

Strips for the user: `runs/smart-reid-shadow/coverage_dev2/review_for_user/`.

## Lock and predeclared fresh-test criteria

The lock was committed before any fresh-bank mining. The full text is in the
lock file.

- **PASS** requires all of:
  - zero verified errors among links coverage_v1 adds or changes relative to v2
    (wrong known player, unknown named, mixed track, bench or non-player);
  - synthetic-unknown false matches no higher than v2;
  - at least 5 added links verified correct.
- **FAIL:** any added verified error.
- **INCONCLUSIVE:** fewer than 5 verified added links, or more than half of the
  added links unverifiable.
- **Coverage target:** at least 10% more verified-correct names than v2.
- **What a pass would mean:** with about 5–15 added links, zero errors bounds
  added-link precision only to roughly 0.74–0.82 (one-sided 95%). A pass rules
  out a gross failure. It does not prove non-inferiority.

## Fresh test

*(filled in after the one-shot run)*

## Reproduce

```bash
PYTHONPATH=src:. $PY -m experiments.teammate_reid.coverage2 extract --banks test_v1 final   # evidence store (GPU)
# development grid: see coverage2_rules.py / coverage2_eval.py; results runs/smart-reid-shadow/coverage_dev2/dev_results.json
PYTHONPATH=src:. $PY -m experiments.teammate_reid.heldout cut --set higher
runs/smart-reid-shadow/run_higher.sh ; runs/smart-reid-shadow/mine_higher.sh
PYTHONPATH=src:. $PY -m experiments.teammate_reid.stage2 --bank higher population
PYTHONPATH=src:. $PY -m experiments.teammate_reid.coverage2 extract --banks higher
PYTHONPATH=src:. $PY -m experiments.teammate_reid.coverage2_fresh decide|queue|score
```
