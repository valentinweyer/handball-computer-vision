# Smart re-ID: fine-tuning plan without player bias (draft v2 for approval, 2026-10-22)

Branch `prototype/smart-reid-shadow`, never merged. **Nothing in this plan runs
until it is approved.** The pass bars (§6) are proposals. Once approved, they
are frozen in a lock file before any training.

Changes from v1:

- **New split.** Training uses only the `train/` videos. Evaluation and test use
  only the `eval/` videos.
- **Test teams never appear in training.** Matches that share teams with
  training are evaluated as **seen teams in a new match**.
- **The rotating team groups are dropped.**

## 1. Goal and condition

**Goal.** Raise smart re-ID recall. In the development data, 63% of the unnamed
player-time belongs to players who have a gallery, but only about 8–10% is
named (`smart-reid-ceiling-2026-10-22.md`). In 81% of those cases the right
player is already ranked first. The gate abstains because the scores of right
and wrong candidates overlap. Better-separated scores are the lever.

**Condition, from you.**

- The model will run on every good-quality handball match, so players from
  training will come back at inference time.
- A model that works better on seen players or teams than on unseen ones is
  biased and is discarded.
- **Parity between seen and unseen teams is the main pass bar.**
- **No test team may appear in training.**

## 2. Design principle: make memorising useless and hard, then measure it

| Measure | What it does |
| --- | --- |
| No identity head | Never classify "this is player 417". |
| Training only within a match | One player at another moment versus his own teammates in the same match. The loss rewards only relative order within the match. |
| No identity across matches | The same player in two matches gets two unrelated labels. |
| Frozen backbone, tiny adapter | Low-rank residual projection on the 512-d SportsReID embeddings, `x + U·Vᵀ·x` at rank 16. About 16k parameters. |
| Anchor to the base model | An extra loss term keeps adapted embeddings close to the original ones. |
| Breadth over depth | Crops per identity are capped. |
| No jersey shortcuts | Strict mask: numbers, names and the torso-and-shorts band hidden. |

## 3. Split (by team, along the video folders)

Teams linked by shared matches decide the split. Training comes only from
`train/`. Everything in `eval/` sharing a team with training is evaluated as
"seen teams, new match". It is never trained on.

**A. Training (adapter), `train/` only:**

| Match | Status |
| --- | --- |
| Balingen–Kiel (BAL–THW) | mined (`dev/`) |
| Berlin–BHC | mined (`dev/`) |
| Lemgo–Magdeburg (TBV–SCM) | mined (`dev_b/`) |
| Göppingen–Leipzig (FAG–LEI) | mined (`dev_b/`) |
| Ludwigshafen–Wetzlar | mined (`fresh/`) |
| Löwen–Flensburg (RNL–SGF) | mined (`fresh/`) |
| Dresden–Melsungen (DRE–MTM) | **not processed yet**; it was the reserve |
| Vinnhorst–Potsdam | **not processed yet**; it was the reserve |

That is 8 matches and 16 teams. For more crops per team, extra windows of these
matches can be processed (about 2–2.5 GPU hours each). That gives more crops
per player, not more teams.

**B. Inner validation (not trained on, not test), `train/`:**

| Match | Role |
| --- | --- |
| Altenholz–Lübeck | mined (`sealed/`) |
| Pforzheim–Bietigheim | mined (`sealed/`) |

These 4 teams are unseen by the adapter. They are used only for the pilot run
and for recalibrating the gate (§7). They never decide the pass bars.

**C. Seen teams in a new match, `eval/`, never trained on:**

| Match | Seen teams | Unseen team |
| --- | --- | --- |
| Kiel–Lemgo | Kiel, Lemgo | n/a |
| Magdeburg–Wetzlar | Magdeburg, Wetzlar | n/a |
| Melsungen–Löwen | Melsungen, Löwen | n/a |
| Gummersbach–Löwen | Löwen | Gummersbach |
| Melsungen–Berlin (not processed) | Melsungen, Berlin | n/a |
| BHC–Frisch Auf (not processed for smart re-ID) | BHC, Göppingen | n/a |

**D. Unseen teams (test), `eval/`, no team in training:**

| Match | Level |
| --- | --- |
| Eisenach–Hamburg | Bundesliga |
| Stuttgart–Erlangen (HC Erlangen) | Bundesliga |
| Saarlouis–Münster | lower league |
| Bielefeld–Aurich | lower league |
| Würzburg–Regensburg | lower league |
| Gummersbach (the team, from C) | Bundesliga |

That is 11 unseen teams, mixed across leagues. All are already processed and
have Stage 2 populations: your labels for Eisenach, Gummersbach and the four
lower-league matches, only assistant labels for Stuttgart.

**Excluded:**

- Balingen II–Erlangen-Bruck. It is a second team of Balingen (training), so
  shared players are possible. It joins D if you confirm the rosters do not
  overlap.
- `e.mp4` in `train/`, which is unidentified. Step 0 checks what it is. It joins
  A only if its teams are disjoint from C and D.

**What this costs in data protection:**

- **The reserve (DRE–MTM, Vinnhorst–Potsdam) becomes training data.** That
  breaks the split fixed on 2026-10-20 and needs your explicit go-ahead. The
  alternative is A without them: 6 matches, 12 teams.
- **No untouched final test remains.** The D matches are unseen by the adapter,
  but the smart re-ID rules were tuned on several of them. A truly fresh system
  test needs new footage (§8).

## 4. Training

- **Inputs.** SportsReID ViT-B/16 embeddings of strict-masked crops, plus 2
  augmented views per crop: mirror, slight crop jitter and slight colour
  jitter. These are computed once on the GPU. Training then runs on cached
  embeddings.
- **Loss.** Supervised contrastive loss within a match: a batch is one match
  with K identities × M crops. Positives are the same identity in another
  fragment at least 2 s apart. Negatives are teammates (full weight) and
  opponents (reduced weight). Plus the anchor term λ·‖f(x) − x‖² and weight
  decay.
- **Time blocks.** The last 1/3 of each training window is never trained on
  (seen evaluation, §5).
- **Hyperparameters fixed in advance:** rank 16, λ, learning rate, epochs, crop
  cap. One pilot run, evaluated only on B and the time blocks. Changes after it
  are allowed only for defects and are recorded. **C and D are not looked at
  before the frozen run.**

## 5. Parity test (core)

**Benchmark.** The established query protocol
(`evaluation.build_queries`/`operating_point`):

- queries are vouched fragments;
- the gallery is same-match fragments, at least 2 s apart;
- 8-crop aggregation;
- synthetic unknowns;
- base model and adapter on identical crops.

**Primary metric M.** Coverage at precision ≥ 0.95 and unknown false-match
rate ≤ 3%, scored per team. Secondary: teammate rank-1.

**Per team, Δ = M(adapter) − M(base):**

| Group | Data | Meaning |
| --- | --- | --- |
| **Unseen** | D (11 teams) | the generalisation we want |
| **Seen, new match** | the seen teams in C | memorisation would show here most strongly |
| Seen, same match | the time blocks of A | control |
| Training material | A's training blocks | memorisation probe, reported only |

**Seen advantage** G = mean Δ(seen, new match) − mean Δ(unseen). The CI comes
from a bootstrap clustered by team.

## 6. Pass bars (proposal, frozen before training)

| # | Condition | Proposal |
| --- | --- | --- |
| P1 | Real gain on unseen teams | mean Δ_unseen > 0 with the lower 95% bound > 0 |
| P2 | Parity | G ≤ max(1 percentage point, ¼·Δ_unseen) **and** the upper 95% bound of G ≤ 4 percentage points |
| P3 | No more false names | Synthetic unknown false-match rate, adapter ≤ base + 1 percentage point, on D |

Missing any one condition means **stop**: report the result and discard the
model.

The limit, stated openly: with 11 unseen and about 9 seen teams in new
matches, only gaps of about 3–4 points or more are detectable. That is why the
P2 bound is 4, not 3 as in v1.

## 7. System check (only if P1–P3 pass)

- **Configuration.** The adapter is plugged into the split_v2 rule as a
  strict-embedding variant. Two variants are fixed in advance:
  - (a) adapted strict score alone;
  - (b) the mean of the raw score and the adapted strict score.
- **Gate.** τ and the margin are recalibrated with the split_v2 calibration
  procedure **on B**, whose teams are unseen by the adapter but are not test
  teams.
- **Evaluation.** On the Stage 2 populations of D, against the existing
  references (your labels, then assistant labels). Newly named segments without
  a reference go into a blind assistant queue with controls, frozen before
  scoring.
- **Bar to lock:**
  - newly correct player-time on D ≥ +25% relative to split_v2;
  - added-link precision ≥ 0.95;
  - synthetic unknown false-match rate ≤ 3%.

## 8. Fresh test (later)

D is unseen by the adapter, but the smart re-ID rules were partly tuned on D.
A truly fresh system test therefore needs **new matches** whose teams appear
neither in A nor in B. The few videos you can still add would be best used
here. Until then the result is "passed on development", not "tested".

## 9. Effort

| Step | Work | Machine |
| --- | --- | --- |
| 0 | Identify `e.mp4`, final team list | CPU, < 1 h |
| 1 | Process DRE–MTM and Vinnhorst–Potsdam, one window each, and mine them | GPU about 5 h |
| 2 | Strict crops and views encoded | GPU, < 1 h |
| 3 | Pilot plus frozen training | minutes |
| 4 | Parity test → stop or continue | CPU, 1 h |
| 5 | System check, blind queue, labels | CPU plus my labels, ½ day |

## 10. Out of scope

- No production change and no merge.
- No backbone training (LoRA would be a separate plan).
- The higher bank appears only in C (Magdeburg–Wetzlar, Melsungen–Löwen) and D
  (Stuttgart–Erlangen), never in training.
- No hyperparameter search on C or D.

## Questions for approval

1. Is the split A–D right? In particular, may the reserve (DRE–MTM,
   Vinnhorst–Potsdam) become training data?
2. Balingen II–Erlangen-Bruck: excluded, or unseen (D)?
3. The pass bars P1–P3 (§6) and the bar to lock (§7)?
4. Labels: blind assistant labels with controls from your labels, your spot
   check optional?
