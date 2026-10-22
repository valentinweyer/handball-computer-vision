# Smart re-ID fine-tune v1: result (2026-10-22)

Branch `prototype/smart-reid-shadow`, never merged.

- Plan: `smart-reid-finetune-plan-2026-10-22.md` (v2, approved).
- Lock: `experiments/teammate_reid/locked/finetune_v1.json`, sha256 `73da2b3991e4…`, tag `smart-reid/finetune-v1-lock`.
- Adapter: sha256 `6754abc6f910…`, `runs/smart-reid-shadow/finetune/adapter.pt`.

## Verdict: FAIL (P2). The adapter is discarded as the lock requires.

The adapter gains clearly on unseen teams, and seen teams in a new match gain
slightly *less*. **The parity bar still fails because the test cannot rule out a
seen-team advantage.** The confidence interval of G is far wider than the 4-point
bound. This is a failure to show parity, not evidence of bias.

| Bar | Condition | Result | Pass |
| --- | --- | --- | --- |
| P1 | Δ_unseen > 0, lower 95% bound > 0 | Δ_unseen = **+0.184**, CI [+0.011, +0.243] | yes |
| P2 | G ≤ max(0.01, ¼·Δ_unseen) **and** upper bound of G ≤ 0.04 | G = **−0.034** (≤ 0.046 holds), CI [−0.210, **+0.162**] | **no** |
| P3 | Unknown FMR on D, adapter ≤ base + 0.01 | 0.0288 vs 0.0288 | yes |

The CIs come from a 2000-replicate bootstrap, clustered by team, with both gates
recomputed in each replicate.

## Groups

M is the coverage of known queries at precision ≥ 0.95 and unknown FMR ≤ 0.03,
using the best two-gate point on the pooled rows of a group.

| Group | Teams | Rows | Base M | Adapter M | Δ | Base rank-1 | Adapter rank-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D, unseen teams | 11 | 1833 | 0.388 | 0.572 | **+0.184** | 0.890 | 0.944 |
| C, seen teams, new match | 7 | 962 | 0.486 | 0.636 | **+0.149** | 0.922 | 0.949 |
| A, seen, same match (last ⅓ of window, never trained) | 16 | 318 | 0.360 | 0.532 | +0.172 | 0.863 | 0.906 |
| A, training material (report only) | 16 | 710 | 0.531 | 0.704 | +0.173 | 0.897 | **0.994** |
| B, inner validation (unseen) | 4 | 208 | 0.304 | **0.245** | **−0.059** | 0.873 | 0.912 |

Notes:

- **Rank-1 on the training material reaches 0.994.** The adapter fits its training
  crops almost perfectly. On new matches, rank-1 gains are the same for seen
  teams (+0.03) and unseen teams (+0.05).
- **B goes the other way: M −0.059 while rank-1 rises.** It is 4 teams and 208
  rows, so it is noisy. Still, of all the groups it is the plainest sign that the
  gain is not uniform. B was the pilot group, so this was visible before the
  frozen run. It is not a defect: the gate grid is built from quantiles of each
  row set's own scores. The frozen run is identical to the pilot (same seed, same
  numbers).

Per-team Δ (`annotations/finetune_v1_parity.json`, `per_team`) ranges from −0.08
to +0.29 in both groups. Individual teams have too few queries for single values
to mean much.

## Why P2 could not pass with this data

G compares 7 seen with 11 unseen teams. Per-team Δ has a standard deviation of
about 0.12 (seen) and 0.13 (unseen), so the bootstrap interval of G spans about
0.37. The plan said openly that only
gaps of about 3–4 points or more are detectable, which assumed a much tighter
spread than the one observed. To get the upper bound below 0.04 at this spread,
we would need several times more teams in both C and D.

## Data and run facts

- **Training crops per match, after the cap of 200 per identity, train block only:**

  | Match | Crops |
  | --- | ---: |
  | Balingen–Kiel | 1153 |
  | Berlin–BHC | 1513 |
  | Lemgo–Magdeburg | 1494 |
  | Göppingen–Leipzig | 1512 |
  | Ludwigshafen–Wetzlar | 1369 |
  | RNL–SGF | 1315 |
  | Dresden–Melsungen | 1140 |
  | Vinnhorst–Potsdam | 755 |

- **Training loss:** 3.04 → 2.56 over 30 epochs.
- **New training windows** (production `process_video` at d0fea4d):
  - DRE–MTM: court in view 82%; 76 vouched fragments, 16 identities, 2354 crops.
  - Vinnhorst–Potsdam: court in view 100%; 31 vouched fragments, 12 identities, 1013 crops.
  - Tracking was interrupted once by a machine restart and resumed at the
    tracking stage.
- **No hyperparameter changes after the pilot.** C and D were evaluated exactly
  once, in `parity`.

## Archived

`experiments/teammate_reid/annotations/`:

- `finetune_v1_parity.json`
- `finetune_v1_pilot.json`
- `finetune_v1_train.json`

Run outputs, logs and embeddings: `runs/smart-reid-shadow/finetune/`,
`runs/smart-reid-shadow/logs/finetune_*.log`.

## What this leaves open

- **The signal is promising, but it is not proof.** On unseen teams M rises
  0.39 → 0.57 at the same false-match rate, and seen teams do not gain more.
- **Showing parity needs more teams.** No hyperparameter or seed change helps
  here: every further run would be another look at C and D. So any second
  attempt must be a new locked plan, with a parity test sized to the observed
  per-team spread. That means more C and D teams, or a different parity design,
  for example paired by team across matches.
- **B moving the other way needs an explanation** before anything would go into
  the system check.
