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
| P3 | Unknown FMR on D, adapter ≤ base + 0.01 | 0.0288 vs 0.0288 | yes, **but vacuous** (see Corrections) |

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
- **B goes the other way: M −0.059 while rank-1 rises.** B was the pilot group,
  so this was visible before the frozen run. The gate grid is built from quantiles
  of each row set's own scores, so the grid is not the cause. The frozen run is
  identical to the pilot (same seed, same numbers). The diagnosis is below (§B
  diagnosis): with each match's own best gate, both B matches improve, but with
  one shared gate the adapter regresses. That is a real calibration weakness.

Per-team Δ (`annotations/finetune_v1_parity.json`, `per_team`) ranges from −0.08
to +0.29 in both groups. Individual teams have too few queries for single values
to mean much.

## Why P2 could not pass with this data

G compares 7 seen team instances (6 distinct teams, see Corrections) with 11
unseen teams. Per-team Δ has a standard deviation of
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
- **B is diagnosed, not cleared** (§B diagnosis):
  - the adapter improves coverage within each match at that match's best gate;
  - with one shared gate it regresses;
  - in Pforzheim, separation from unknowns gets worse.

  Calibration across matches has to be solved before any system use.
- **A failed further attempt on this footage would mean "insufficient evidence;
  stop reusing C and D"**, not "fine-tuning is a dead end".

## Corrections (after two external reviews, 2026-10-22)

The reviews found flaws in the v3 plan's evaluation design and in my first B
write-up. The flaws below apply to this v1 result. The historical FAIL stands.

1. **P3 was vacuous.**
   - M only admits gates with unknown FMR ≤ 0.03. The bar allowed base + 0.01 =
     0.0388, so any qualifying adapter gate passes whenever base FMR is at least
     0.02.
   - Both came out at 27/936 = 0.0288. That tie is **not** the largest rate the
     unknown count allows: 28/936 = 0.0299 also meets the limit. My first
     explanation of it was wrong.
   - P3 says nothing about false names. An honest version fixes the gates
     elsewhere, for example on B, and measures FMR on D at those gates.
2. **The bootstrap clusters were wrong.**
   - The team key is `match|cluster`, so Melsungen–Löwen and Gummersbach–Löwen
     are separate clusters. The seen group is 6 distinct teams, not 7.
   - Match-level dependence was ignored, because both teams of a match share
     conditions.
   - The CI of G is therefore unreliable. How it would change after correction is
     not established: M is a nonlinear best-gate statistic.
3. **M is a best-gate statistic on the evaluation rows.** It measures achievable
   separation, not the performance of a gate calibrated elsewhere. It cannot be
   averaged over teams or folds.
4. **Training and evaluation positives differ slightly.**
   - Training allows positives from another fragment of the *same track*, while
     evaluation excludes the query's own track.
   - They are 34 of 745 same-identity fragment pairs in A's train block (5%).

## B diagnosis (B rows only, `finetune diag_b`, `annotations/finetune_v1_diag_b.json`)

B has 102 known and 106 unknown queries (102 of them synthetic).

| | Base | Adapter |
| --- | ---: | ---: |
| M, one shared gate over both matches | 0.304 | 0.245 |
| Correct accepted queries at that gate (3 unknown false accepts in both) | 31 | 25 |
| M, precision bar only | 0.804 | 0.882 |
| M, FMR bar only | 0.304 | 0.245 |
| Rank-1 | 0.873 | 0.912 |
| Score AUC, right against unknown, pooled | 0.770 | 0.809 |
| Margin AUC, right against unknown, pooled | 0.690 | 0.670 |
| Margin AUC, right against wrong, pooled | 0.857 | 0.955 |
| M, Altenholz–Lübeck, own best gate | 0.615 | 0.718 |
| Score AUC, Altenholz, right against unknown | 0.922 | 0.958 |
| Margin AUC, Altenholz, right against unknown | 0.687 | 0.728 |
| M, Pforzheim–Bietigheim, own best gate | 0.206 | 0.270 |
| **Score AUC, Pforzheim, right against unknown** | 0.790 | **0.772** |
| **Margin AUC, Pforzheim, right against unknown** | 0.714 | **0.648** |

**Conclusion.** The adapter improves coverage within each match at that match's
own optimised gate. With one shared gate on B it regresses, from 31 to 25 correct
accepts at the same 3 false accepts. In Pforzheim it separates unknowns *worse*
on both score and margin.

1. **FMR binds, precision does not.** With 106 unknowns, FMR ≤ 3% allows 3 false
   accepts, so pooled M is set by the **4th-highest-scoring unknown**.
2. **Those top unknowns all come from Pforzheim.** It has the higher similarity
   level:
   - Pforzheim's median unknown score (base 0.764) is above Altenholz's median
     correct score (0.742).
   - The adapter lifts Altenholz's correct scores (0.742 → 0.810) and Pforzheim's
     top unknowns (0.826 → 0.865).
   - The shared gate moves from 0.798 to 0.852.
3. **Calibration across matches is a real deployment problem.** Production uses one
   gate for every video. Per-match best gates show potential, not a solution. A
   gate fixed on B makes the evaluation more honest but does not fix this.
4. **Overall AUC is not the operating region.** The relevant region is the
   top-scoring unknowns near FMR 3%. Pooled AUC improves while that tail gets
   worse.

### Duplicate-identity audit of Pforzheim's top unknowns

Records are in `annotations/finetune_v1_b_dup_audit.json`. The strips are in
`runs/smart-reid-shadow/finetune/audit_b/`.

The question was whether the high-margin unknowns are one player mined under two
identities.

- **Method.** Take the union of the 6 highest-scoring unknowns under each model,
  excluding queries vetoed by the halves guard: 9 queries.
- **Temporal check.** Were the query identity and the best candidate on court at
  the same time?
- **Visual check.** 8 crops of the query against 8 gallery crops, raw (unmasked)
  so the numbers can be read.

| Query | Best candidate | Score base → adapter | Margin base → adapter | On court together | Visual |
| --- | --- | --- | --- | ---: | --- |
| team 0 #10 (3 queries) | #36 | 0.82–0.83 → 0.84–0.87 | 0.09 → 0.12–0.13 | 1300 frames | different players: #10 has longer dark hair; #36 short hair and a black arm sleeve. Same kit, both wear black knee sleeves |
| team 0 #36 (2 queries) | #10 | 0.80–0.82 → 0.86 | 0.09–0.13 → 0.16–0.17 | 1300 frames | same pair as above |
| team 1 #15 (3 queries) | #20 (under base, one query's best was #44) | 0.77–0.81 → 0.84–0.85 | 0.00–0.01 → 0.01–0.03 | 1580 frames | different: #15 blond and slim; #20 dark short hair, stockier |
| team 1 #15 (1 query) | #4 | 0.79 → 0.85 | 0.01 → 0.02 | 1250 frames | different: #4 has a ponytail |
| team 1 #20 (1 query) | #15 | 0.80 → 0.84 | 0.01 → 0.02 | 1580 frames | same pair as #15/#20 |

**Result: no duplicate identity.** Every pair was on court together, and every pair
is visibly two players.

- **The hypothesis is rejected.** No label correction is needed.
- **The top unknowns are genuine look-alike teammates.** The strict mask hides
  numbers and the torso band, which leaves only hair, arms, legs and build.
- **The adapter makes the worst pair more confusable.** For #10/#36 it raises both
  the score *and* the margin of the wrong teammate (margin 0.09–0.13 →
  0.12–0.17). That is the opposite of what within-match teammate training should
  do.
- **Two look-alike pairs set the B gate.**
- **Side finding, not a label defect:** #20 (frames 6511–7090) and #15 (frames
  7271–7520) share tracker id 173, an identity switch within one track. Fragment
  splitting assigned both pieces to the right player.
