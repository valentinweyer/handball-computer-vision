# Smart re-ID: a guard against boxes that switch person (2026-10-21, development, negative)

Branch `prototype/smart-reid-shadow`, never merged. This uses development data
only: the two spent Stage 2 stores (`test_v1`, `final`). The higher-league
bank's two mixed errors motivated the question but were not used: that bank
stays test data.

**Question.** Can a guard stop the matcher naming segments whose box switches
to another person (mixed tracks), mainly short tail switches, without losing
correct names? The half-split check already in the rules misses these. The
round-2 `mixed_guard` needs both people to hold at least 25% of the crops, so it
cannot see a short tail by design.

**Data.** All labelled development segments:

- **17 mixed.** 4 human labels and 13 assistant labels, several of them
  "another player in front" or "team huddle".
- **224 with a readable number.**

Signals are computed from the stored query embeddings, the crop frames and the
bundle boxes.

## Signals tried

Each threshold sets how many mixed segments a signal catches and how many
correctly labelled segments it flags along with them.

| signal | mixed caught | correct segments flagged |
| --- | ---: | ---: |
| appearance change point in time (best split, contrast) | 4 / 15 | 21 / 206 (10%) |
| end-block contrast (first/last sixth vs rest) | 0 / 15 | 11 / 206 |
| appearance before vs after each box crossing (another box covers ≥ 50%) | 8 / 17 | 34 / 224 (15%) |
| same, at a looser threshold | 13 / 17 | 81 / 224 (36%) |
| named player's own gallery score, before vs after each crossing | 3 / 17 | 11 / 195 (6%) |

Crossings are real in the data: mixed segments average 6.5 crossing episodes
against 3.5 for correct ones. But appearance across a crossing changes about as
much for a correct segment (pose, occlusion, motion blur) as for a switch.

**Decision-level arithmetic.** Under coverage_v1, development names include 1
mixed track against about 120 correct names. A guard that flags 6–15% of correct
segments would remove 7–18 correct names to stop at most 1 error. None of these
signals is close to worth it.

The classic tail case shows the limit. In Kiel S259 the box holds #4, then #23
only in the last frame. With crops sampled every 10 frames, the tail is at most
one crop, which no appearance statistic can tell from a single occluded crop.

## Verdict

**No guard. The locked rules are unchanged.** Short tail switches stay a known
error class:

- they are reported as errors in every evaluation;
- they are the reason a mixed segment counts as wrong even when the named
  player fills most of it.

Joining segments by motion (the planned next step) was **not started**. It
would lengthen segments across exactly these crossings, and there is now no
guard to catch the switches it would create.

Ideas that might work, not tried:

1. **Trim instead of abstain.** Name only up to the last crossing before a
   detected change. This changes the unit being labelled, so the reference
   labels would need per-frame spans.
2. **Denser crops near crossings.** Encode every frame for a second around
   each crossing, so that a one-crop tail becomes several crops.
3. **Fix it upstream in the tracker**, where the switch happens, rather than in
   re-ID.
