# Smart re-ID: how much of the unnamed time could a matcher name? (2026-10-22, development)

Branch `prototype/smart-reid-shadow`, never merged. Development data only:
the seven Stage 2 populations (`test_v1`: Kiel, Eisenach, Gummersbach;
`final`: Saarlouis, Bielefeld, Balingen II, Würzburg). No test match was
touched.

**Question.** Is smart re-ID limited by missing galleries, or by the matcher?
This decides whether fine-tuning the appearance model is worth it.

## Method

- **Population.** 971 segments with 14,864 s of player-time the production
  pipeline left unnamed.
- **Sampling.** The Stage 2 queues hold every segment split_v2 named (90,
  fully labelled), plus a seeded random sample of 40 per bank from the rest
  (`annotations/sealed/stage2_*_queue_membership_SEALED.json`). You
  blind-labelled these as part of Stage 2. The remainder is estimated from the
  random sample, weighted by unnamed seconds. The 95% CI comes from a
  bootstrap over the sample.
- **Reachable** means the true player, by your label, has a gallery identity in
  the window (or, separately, in the extra windows).

## Where the unnamed time goes

| category | share of unnamed player-time | 95% CI |
| --- | ---: | --- |
| **true player has a gallery in the window** | **63%** | 49–76% |
| true player has a gallery only in the extra windows | 1.4% | 0–5% |
| true player has no gallery | 14% | 5–26% |
| not a player (referee, staff, false box) | 12% | 5–22% |
| unreadable for the labeller | 5% | 1–10% |
| mixed track | 4% | 0–10% |

Today the system names correctly 7.9% (split_v2) to 9.7% (coverage_v1) of
this time. **The ceiling for a perfect matcher is about six times higher.**
The limit is the matcher, not gallery availability. Extra footage barely
changes the ceiling, by 1.4 points.

This corrects my earlier statement that most unnamed segments have nothing to
match against. That was wrong.

## Why the matcher abstains

Take the 48 random-sample segments whose player has a window gallery:

- **39 (81%) already have the right player ranked first.** For 4 the right
  player is ranked 2nd or 3rd, for 1 lower. For 4 the right player was removed
  by co-presence or reads.
- On those 39, v2 abstains:
  - 22 times because the score is below the gate;
  - 7 times because only one candidate is left;
  - once because the two halves of the segment disagree.

  9 are named. Margins are rarely the problem: 31 of 32 margins are above the
  locked margin.

**Why the gate cannot simply be lowered.** If the top-1 candidate were named
for all 80 random segments:

- 39 would be correct;
- 18 would be a wrong or unknown player;
- 11 would name a non-player;
- 3 would be mixed.

Top-1 scores overlap:

- correct: 0.62–0.85;
- wrong or unknown: 0.51–0.77;
- the gate τ is 0.784.

Above about 0.77 the sample has no wrong player. Below that, correct and wrong
are about equally frequent. Recall can only rise if the scores separate correct
from wrong better. That is exactly what fine-tuning the appearance model
targets.

## Caveats

- 80 random segments; the CIs are wide.
- Reachable is necessary, not sufficient. Many reachable segments are short,
  distant or occluded, and no matcher will name all of them.
- The top-1 scores of non-player and mixed segments are not broken out above.
  A lower gate would also have to reject those.

## Reproduce

```bash
PYTHONPATH=src:. $PY -m experiments.teammate_reid.ceiling   # CPU, minutes -> runs/smart-reid-shadow/ceiling/ceiling.json
```
