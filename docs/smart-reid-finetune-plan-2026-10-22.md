# Smart re-ID: fine-tuning plan without player bias (draft for approval, 2026-10-22)

Branch `prototype/smart-reid-shadow`, never merged. **Nothing in this plan runs
until it is approved.** The pass bars in §6 are proposals. Once approved, they
are frozen in a lock file before any training.

## 1. Goal and condition

**Goal.** Raise smart re-ID recall. In the development data, 63% of the unnamed
player-time belongs to players who have a gallery, but only about 8–10% is
named (`smart-reid-ceiling-2026-10-22.md`). In 81% of those cases the right
player is already ranked first. The gate abstains because the scores of right
and wrong candidates overlap. Better-separated scores are the lever.

**Condition, from you.** The model will run on every good-quality handball
match, Bundesliga and lower leagues, so players from training will come back
at inference time. A model that works better on seen players or teams than on
unseen ones is biased and is discarded. **Parity between seen and unseen teams
is the main pass bar, not a side check.**

## 2. Design principle: make memorising useless and hard, then measure it

| Measure | What it does |
| --- | --- |
| No identity head | Never classify "this is player 417". |
| Training only within a match | One player at another moment versus his own teammates in the same match. The loss rewards only relative order within the match. |
| No identity across matches | The same player in two matches gets two unrelated labels. |
| Frozen backbone, tiny adapter | Low-rank residual projection on the 512-d SportsReID embeddings, `x + U·Vᵀ·x` at rank 16. About 16k parameters, too few to store hundreds of individuals. |
| Anchor to the base model | An extra loss term keeps adapted embeddings close to the original ones, so the change stays a small, general correction. |
| Breadth over depth | Crops per identity are capped, so no player dominates. |
| No jersey shortcuts | Input crops use the **strict mask**: numbers, names and the torso-and-shorts band hidden (as in the jersey-independence rule). What remains is head, limbs, build and shoes. |
| Rotating held-out teams | The parity test, §5. |

## 3. Data

**Training pool: development only.** 15 matches, all mined automatically
(number-vouched fragments):

- the 8 split-v1 tuning matches:
  - Balingen–Kiel
  - Berlin–BHC
  - Lemgo–Magdeburg
  - Göppingen–Leipzig
  - Ludwigshafen–Wetzlar
  - RNL–SGF
  - Altenholz–Lübeck
  - Pforzheim–Bietigheim
- Kiel–Lemgo (56:00 and 76:00)
- Eisenach–Hamburg (3 windows)
- Gummersbach–Löwen (3 windows)
- Saarlouis–Münster (3 windows)
- Bielefeld–Aurich (3 windows)
- Balingen II–Erlangen (3 windows)
- Würzburg–Regensburg (3 windows)

That is about 25 windows, each with 5–18 vouched identities per match window.

**Excluded from training:**

- **The higher bank** (Melsungen–Löwen, Magdeburg–Wetzlar, Stuttgart–Erlangen).
  It stays test data. It also **cannot serve as an unseen test**: Magdeburg,
  Wetzlar and Löwen all play in development matches.
- **The reserve** (DRE–MTM, Vinnhorst–Potsdam): the final test, §8.

**Two time blocks per window:**

- the first 2/3 of each window is **training**;
- the last 1/3 is **seen evaluation**: the same players, never trained on.

**Team groups.** Teams that appear in several matches link those matches into
one block:

- Kiel / Lemgo / Balingen / Magdeburg: Balingen–Kiel, Kiel–Lemgo,
  Lemgo–Magdeburg, and Balingen II–Erlangen as a precaution;
- Löwen: RNL–SGF, Gummersbach–Löwen.

The remaining 9 matches stand alone. The blocks are distributed over **5
groups** of about equal size. A team is always entirely in one group. Step 0
checks the complete team list, including whether "MTM" from the reserve appears
in development.

## 4. Training

- **Inputs.** SportsReID ViT-B/16 embeddings of strict-masked crops, plus 2
  augmented views per crop: mirror, slight crop jitter and slight colour
  jitter. These are computed once on the GPU, under an hour. After that,
  training runs on cached embeddings in minutes.
- **Loss.** Supervised contrastive loss within a match: a batch is one match
  with K identities × M crops. Positives are the same identity in another
  fragment at least 2 s apart. Negatives are teammates (full weight) and
  opponents (reduced weight). Plus the anchor term λ·‖f(x) − x‖² and weight
  decay.
- **Hyperparameters fixed in advance.** Rank 16, λ, learning rate, a fixed
  number of epochs, crop cap per identity. **No search over the evaluation
  groups.** One pilot run on group 1 only checks that the pipeline works.
  Changes after it are allowed only for defects, are recorded, and group 1 is
  then trained again from scratch.
- **5 models.** Model *k* never sees group *k*.

## 5. Parity test (core)

**Benchmark.** The established query protocol
(`evaluation.build_queries`/`operating_point`):

- queries are vouched fragments;
- the gallery is same-match fragments, at least 2 s apart;
- 8-crop aggregation;
- synthetic unknowns (own identity removed);
- every condition is scored with the base model and with the adapter, on
  identical crops.

**Primary metric M.** Coverage at precision ≥ 0.95 and unknown false-match
rate ≤ 3%. This measures the separation of right and wrong, which is the
bottleneck. Teammate rank-1 is secondary.

**Per team:**

- Δ_unseen = M(adapter) − M(base), measured with the model that did not see
  this team;
- Δ_seen = the same on the team's seen-evaluation block, measured with the
  4 models that trained on the team (averaged);
- **seen advantage G = Δ_seen − Δ_unseen.**

Every team is unseen exactly once, so the comparison uses all teams. The CIs
come from a bootstrap clustered by team.

**Memorisation probe.** Also report Δ on the teams' own training blocks. A
large lead of training block over seen evaluation, or of seen over unseen,
signals storage instead of skill.

## 6. Pass bars (proposal, frozen before training)

| # | Condition | Proposal |
| --- | --- | --- |
| P1 | Real gain on unseen teams | Δ_unseen > 0 with the lower 95% bound > 0 |
| P2 | Parity | G ≤ max(1 percentage point, ¼·Δ_unseen) **and** the upper 95% bound of G ≤ 3 percentage points |
| P3 | No more false names | Synthetic unknown false-match rate, adapter ≤ base + 1 percentage point, on unseen teams |

Missing any one condition means **stop**: report the result, discard the
model, and stop the fine-tuning work.

The limit, stated openly: with about 40 teams, gaps below about 2–3 points
cannot be detected. "Parity passed" therefore means "no gap above that size".

## 7. System check on development (only if P1–P3 pass)

- **Configuration.** The adapter is plugged into the split_v2 rule as a
  strict-embedding variant. Two variants are fixed in advance:
  - (a) adapted strict score alone;
  - (b) the mean of the raw score and the adapted strict score.
- **Gate.** τ and the margin are recalibrated with the split_v2 calibration
  procedure. The score scale changes, so the old gate does not fit.
- **Clean evaluation.** Each development match is scored with the model that
  did not see its teams (out-of-fold).
- **Evaluation.** On the 7 Stage 2 populations, against the references
  (your labels, then earlier assistant labels). Newly named segments without a
  reference go into a blind assistant queue with controls, frozen before
  scoring.
- **Bar to lock:**
  - newly correct player-time ≥ +25% relative to split_v2;
  - added-link precision ≥ 0.95;
  - synthetic unknown false-match rate ≤ 3%.

## 8. Final test (only after the lock, with your approval)

- **Final model.** The adapter trained on all development teams with the same
  recipe. The parity guarantee covers the recipe, not this specific model.
- **Test data.** The reserve, DRE–MTM and Vinnhorst–Potsdam. They are unseen by
  construction and are spent by this test. They are processed with production
  `process_video` first, if not done yet, at about 2–2.5 GPU hours per window.
- **Verdict.** The established fresh-test rule:
  - PASS / FAIL / INCONCLUSIVE;
  - at least 5 verified added links;
  - no added error;
  - synthetic FM not worse.

## 9. Effort

| Step | Work | Machine |
| --- | --- | --- |
| 0 | Team list, groups, data inventory | CPU, 1 h |
| 1 | Strict crops and views encoded | GPU, < 1 h |
| 2 | 5 adapters plus pilot | minutes |
| 3 | Parity test → stop or continue | CPU, 1 h |
| 4 | System check, blind queue, labels | CPU plus my labels, ½ day |
| 5 | Reserve processing and test | GPU 5–10 h plus labels |

## 10. Out of scope

- No production change and no merge.
- No backbone training: LoRA or full fine-tuning would be a separate plan, only
  if the adapter passes and the gain is too small.
- No use of the higher bank for tuning or testing.
- No hyperparameter search on evaluation groups.

## Questions for approval

1. The pass bars P1–P3 (§6) and the bar to lock (§7): accepted as written?
2. Spend the reserve on the final test (§8)?
3. Labels: blind assistant labels with controls from your labels, and your spot
   check optional, as in the extra-galleries work?
