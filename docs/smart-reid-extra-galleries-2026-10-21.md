# Smart re-ID: galleries from more of the match (2026-10-21, development)

Branch `prototype/smart-reid-shadow`, never merged. This is development data
only: Kiel–Lemgo and Eisenach–Hamburg, both spent banks. No test match was
touched.

**Question.** If a match contributes more footage than the 10-minute window
being named, do more players get a gallery, and do more unresolved segments get
a correct name without new errors?

## Setup

**Population.** The unchanged Stage 2 population of the two 66:00 windows:
223 segments, 2,722 s.

**Extra footage.** Four windows next to the 66:00 windows, none overlapping it:

| Match | Earlier window | Later window |
| --- | --- | --- |
| Kiel | 56:00 | 76:00 |
| Eisenach | 57:00, 9 min (second half starts about 56:40) | 76:00 |

All were processed with production `process_video` and mined automatically.
They supply gallery fragments only, never population segments.

**Volume added.**

| Match | Vouched fragments | Crops | New players | Players with a bigger gallery |
| --- | ---: | ---: | --- | ---: |
| Kiel | 152 | 4,826 | #91, #11, #24, #48 | 16 |
| Eisenach | 145 | 3,499 | blue #14, red #11 | 18 |

**Team alignment.** Team ids are anonymous clusters per video. Each extra
window's clusters were mapped to the 66:00 clusters by gallery torso colour,
without using labels. All four windows mapped one to one.

**Rules.** The split_v2 and coverage_v1 locks, unchanged
(`extra_galleries.py`).

**Gallery size.** Top-3 scoring over a bigger gallery raises every candidate's
score, while the gate was set on 10-minute galleries. So every comparison is
also run with a fixed crop budget per player (cap 150 or 100 crops, spread over
match time). Without the cap, base and extra cannot be compared like for like.

**Reference.** Your labels and adjudications where they exist, then the earlier
assistant labels. The 21 newly named segments no reference covered went into a
new blind assistant queue:

- shuffled codes;
- full-resolution strips with team reference crops (the fix from the fresh
  test);
- 8 controls from your labels mixed in;
- frozen as `annotations/extra_galleries_assistant_labels_v1.json` (sha256
  `28c85af9…`) before any comparison.

On the controls, I agree with you on all 3 I could read. I could not read 4. On
one (S268), you labelled #61 and I flagged a possible box switch; your label
stands.

## Results

Errors are wrong player + unknown named + mixed. "Synthetic FM" means the truth
player's gallery was removed and the rule named someone anyway. Criterion v2
allows at most 3%.

| gallery | rule | correct | errors | precision | newly correct time | synthetic FM | unreadable |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 66:00 only | v2 | 50 | 3 | 0.943 | 534 s | 2/56 | 25 |
| + extra, uncapped | v2 | **67** (+34%) | 3 | 0.957 | 814 s | **7/56** | 42 |
| 66:00 only | coverage_v1 | 58 | 3 | 0.951 | 671 s | 2/56 | 28 |
| + extra, uncapped | coverage_v1 | **79** (+36%) | 4 | 0.952 | 987 s | **7/56** | 46 |
| 66:00 only, cap 150 | v2 | 47 | 3 | 0.940 | 510 s | 1/56 | 24 |
| + extra, cap 150 | v2 | 58 (+23%) | 3 | 0.951 | 761 s | 3/56 | 35 |
| 66:00 only, cap 100 | v2 | 46 | 2 | 0.958 | 502 s | 1/56 | 24 |
| + extra, cap 100 | v2 | 53 (+15%) | 3 | 0.946 | 690 s | 2/56 | 32 |
| + extra, cap 100 | coverage_v1 | 63 | 3 | 0.955 | 857 s | 2/56 | 35 |

### What more of the match buys

- **More correct names, no new wrong player.**
  - Under v2, more names are correct: +17 uncapped, +11 at cap 150, +7 at
    cap 100.
  - Newly correct player-time rises 50% uncapped (534 s → 814 s).
  - Every added name was verified correct or unreadable. None is a wrong known
    player and none is an unknown named.
  - The one added error is under coverage_v1: Kiel S039, a known mixed segment
    (`coverage_dev` labels), named #6.
  - At cap 150, one correct v2 name (Eisenach S139, #33) is lost: the bigger
    rival galleries closed its margin.
- **A player the 66:00 window could not know gets named.** Kiel #91 has no
  vouched fragment in the 66:00 window, so every #91 segment there was a real
  unknown. Both rules correctly abstained on them. With the extra windows:
  - 5 segments are named #91 and verified correct, by clear 91s on back or
    chest;
  - 5 more are named #91 and unreadable;
  - none is a wrong #91.

  This is the "a player seen reliably in minute 55 gets a name in minute 66"
  effect you predicted.

### What it costs

- **Synthetic unknowns get matched more often.** False matches rise from 2/56 to
  7/56 (12.5%) uncapped, well above the 3% criterion. Seven players had their
  own gallery removed and still got a confident name. The extra windows give
  every teammate more look-alike crops, and the gate was calibrated for
  10-minute galleries.
- **The crop cap only partly helps.** At cap 100, false matches are 2/56, as in
  the 66:00 base. But the gain is down to +7 correct names, and there is one
  more mixed name than base at the same cap. The cap shows that much of the
  uncapped gain comes from score inflation, not only from better views.

## Verdict (development only)

**More footage per match is a real recall lever.**

- Verified-correct names rise 15–34% depending on gallery budget.
- Players absent from the window get a gallery.
- No wrong known player was introduced.

**It is not lockable as is.** With the locked gate, unknown players get named
far too often. The gate assumes the 10-minute gallery size it was set on.

Next, still on development data:

1. **Make the score independent of gallery size.** For example, average the
   score over fixed-size subsets drawn evenly over match time, or calibrate the
   gate per gallery size with leave-one-out on the galleries themselves. That is
   a method choice, not per-match tuning.
2. **Then re-measure synthetic FM** with all extra footage. The aim is to keep
   the #91-type gains while bringing false matches back to the 66:00 rate.
3. **Extract only the extra windows' galleries.** In production that means
   processing more of each match. Once one 10-minute window costs about
   2–2.5 GPU hours, the runtime cost has to be weighed.

## Data and reproduce

```bash
runs/smart-reid-shadow/run_extra_windows.sh     # cut + production process_video (done)
runs/smart-reid-shadow/mine_extra_dev.sh        # mine, features, sheets -> runs/smart-reid-shadow/extra_dev/
PYTHONPATH=src:. $PY -m experiments.teammate_reid.extra_galleries   # grid -> runs/smart-reid-shadow/extra_galleries/result.json
```

Labels, queue and a summary of the result are archived in
`experiments/teammate_reid/annotations/extra_galleries_*`. The extra windows
are registered as `heldout.EXTRA_DEV_WINDOWS` (`run.py --set extra_dev`).
