# Smart re-ID: raising coverage on unresolved segments (2026-10-20, development)

Branch `prototype/smart-reid-shadow`, never merged.

- **Data.** Everything here uses the two spent Stage 2 banks: Kiel, Eisenach
  and Gummersbach (test_v1), plus the four lower-league windows (final). The
  user released them as development data on 2026-10-20, so nothing here is a
  test.
- **Question.** Can more of the segments the production pipeline left
  unnamed get a name, without losing precision?

## Method

1. **Evidence cache** (`coverage.py extract`, one GPU pass of about 15 min).
   For all 971 population segments the cache stores:
   - up to 60 crops each, encoded with the locked encoder;
   - every candidate under the split-v2 gallery protocol, with per-crop scores;
   - the evidence behind each veto: co-present frames, duplicate-box frames
     (IoU ≥ 0.5 with the segment's own box), and the candidate's nearest gallery
     positions.

   The locked v2 rule, re-run from this cache, reproduces the final bank's
   sealed decisions exactly (0 of 663 differ).
2. **Rule variants** (`coverage_rules.py`), evaluated offline:
   - the locked gate;
   - accepting a lone candidate above a score;
   - softer co-presence vetoes;
   - a stricter number-contradiction check;
   - a speed veto;
   - team guards;
   - gate grids;
   - two-pass chaining: names from pass 1 join their player's gallery and veto
     that player on overlapping segments in pass 2.
3. **Reference.** Your labels, meaning blind labels plus your adjudication
   (`score_v3`), wherever a segment was labelled. Every other segment that a
   candidate rule named (137) went into a blind assistant queue:
   - full-resolution strips, only the target box drawn, shuffled codes;
   - 16 controls with human labels mixed in;
   - frozen as `annotations/coverage_dev_assistant_labels_v1.json` (sha256
     `4bdacfdd…`) before any comparison. These are assistant labels, not human
     ground truth.

   On the controls: 7 agree with you, 8 were unreadable to me, and 1 disagrees,
   S241 (see open points).

## Results: every name each rule gives, checked against the reference

Population: 971 segments, 14,896 s.

- *unreadable*: my blind strip could not confirm the number. Those names are
  unverified, neither right nor wrong.
- *mixed*: the tracked box holds two people.

| rule | named | correct | wrong | mixed | unreadable | precision (correct / (correct + wrong)) | correct, share of segments / seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| locked v2 | 138 | 100 | 3 | 1 | 26 | 0.971 | 10.3% / 7.9% |
| **safe** = v2 + lone candidate ≥ 0.80 + duplicate boxes no longer veto | 161 | **119** | **3** | 1 | 30 | **0.975** | **12.3% / 9.7%** |
| safe + chaining (same gate in pass 2) | 199 | 140 | 5 | 6 | 40 | 0.966 (0.927 counting mixed as wrong) | 14.4% / 11.2% |
| safe + chaining, pass 2 at τ 0.76, margin 0.045 | 248 | 167 | 8 | 10 | 55 | 0.954 (0.903 counting mixed) | 17.2% / 13.9% |

The correct shares are lower bounds, because unreadable names count as not
correct.

**safe adds 19 correct names (+19%) and no new error.** Both changes are
narrow:

- **Lone candidate.** When vetoes leave exactly one candidate, the matcher has
  always abstained. It now names that candidate if the score is ≥ 0.80.
- **Duplicate boxes.** The co-presence veto no longer fires on frames where the
  "other" box showing the candidate's number overlaps the segment's own box
  (IoU ≥ 0.5). That is the same person seen by two tracks, not a second player.

**Chaining adds more but brings in mixed segments.** Its extra names are
mostly correct (+21 at the same gate), but they include 5 more segments
whose box switches person, which the half-split guard does not catch. Pass 2
at a lower gate pushes the reviewed precision near or below 0.95. Neither is
ready.

## Levers that did not help

- **Stricter number contradiction** (one read is enough): removes 20 correct
  names and no wrong one. Single reads are too noisy.
- **Speed veto** (implausible movement between gallery fragment and segment):
  never fired. Court positions at fragment ends are often off court, so it has
  too little valid evidence.
- **Co-presence thresholds by frame count:** no gain, because the true vetoes
  are long overlaps.
- **Team guards** (abstain on an unsettled team, or on thin team galleries):
  they cost more right names than they save wrong ones.
- **Lowering the gate alone** (τ 0.70–0.76): coverage rises to 22–30%, but
  real unknowns get named. On your labels, 2–5 players with no gallery got a
  confident name, and the estimated precision falls to 0.87–0.93. Scores do
  not separate a real unknown from a look-alike teammate well enough, and
  per-candidate score normalisation did not change that.

## Errors found (all rules)

| segment | named | reference | source |
| --- | --- | --- | --- |
| Balingen II S158 | #10 | #7 | human, the known teammate confusion |
| Eisenach S000, S011 | Eisenach blue #99 | #55, read on the shorts | assistant. The locked v2 rule names these; split v1 never did, so they are new. Worth a human look. |
| Kiel S259 | #4 | mixed (#4, then #23) | assistant |

Chaining adds S039, S195, S214, S235, S241, S319 and S328. Most of those are
mixed segments.

## Open points

1. **S241 (Bielefeld):** you labelled it #7, but chaining names it #6, and my
   blind read (code Q044) also shows a 6 on the chest. Please check it. If it
   is #6, it is a label error like S449 and S278.
2. **S000 and S011 (Eisenach, #99 against #55):** these are matcher errors
   only if my reading of the shorts is right.
3. **Recommendation: lock "safe" as split v3 for the higher-league bank.** It
   is two narrow changes and showed no added error on development data. Lock
   only after you agree, because a lock is never rewritten.
4. **Chaining needs a mixed-segment guard before it is worth testing.** The
   current half-split check misses switches inside one half.

## Reproduce

```bash
PYTHONPATH=src:. $PY -m experiments.teammate_reid.coverage extract    # GPU, ~15 min
PYTHONPATH=src:. $PY -m experiments.teammate_reid.coverage evaluate   # offline grid
```

Outputs are in `runs/smart-reid-shadow/coverage_dev/`: `*_evidence.json`,
`rules.json`, `verify/` (queue, pages, labels) and `verified_rules.json`.
