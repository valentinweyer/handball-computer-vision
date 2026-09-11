"""Does a partial read catch the trailing digit? Measure it on the labelled set.

`_Votes.resolved_counts` folds a 1-digit read into a 2-digit value that ends
with it, on the ground that a crop catching only the trailing digit of "17"
reads as "7" while the reverse does not occur. The rule was tuned on docTR's
reader; this re-measures its premise on the reader that ships.

Reads the pad-0 predictions the crop-padding sweep already stored, so it needs
no GPU and no second inference pass:

    python -m scripts.audit_fold_rule \
        --sweep runs/number_eval_1080p/crop_padding_sweep_baudm.json \
        --labels data/annotations/jersey/number_eval_1080p_labels.json

The gate sweep is the point of the whole script. One-sided folding is safe
because every leading-digit read measured falls below PARSEQ_MIN_CONFIDENCE --
a dependency on the gate, not a property of the reader.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from handball_cv.jersey.identity import PARSEQ_MIN_CONFIDENCE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sweep", type=Path, required=True)
parser.add_argument("--labels", type=Path, required=True)
parser.add_argument("--confidence", type=float, default=PARSEQ_MIN_CONFIDENCE)
args = parser.parse_args()

sweep = json.loads(args.sweep.read_text())
labels = json.loads(args.labels.read_text())["labels"]
raw = next(r for r in sweep["pads"] if tuple(r["pad"]) == (0.0, 0.0))["raw"]

CONF = args.confidence
truth = {int(i): v["value"] for i, v in labels.items()
         if v.get("status") == "readable" and v.get("value")}

pred = {}
for i, r in raw.items():
    t = str(r["text"]).strip()
    if r["confidence"] >= CONF and t.isdigit() and 1 <= len(t) <= 2:
        pred[int(i)] = t

pairs = [(truth[i], pred[i]) for i in truth if i in pred]
print(f"readable crops {len(truth)} | gated reads {len(pairs)} (conf>={CONF})\n")

two = [(t, p) for t, p in pairs if len(t) == 2]
one = [(t, p) for t, p in pairs if len(t) == 1]

# (a)/(b): what does a 2-digit jersey read as, when it reads as one digit?
short = [(t, p) for t, p in two if len(p) == 1]
trailing = [(t, p) for t, p in short if p == t[1]]
leading  = [(t, p) for t, p in short if p == t[0] and p != t[1]]
neither  = [(t, p) for t, p in short if p != t[0] and p != t[1]]
print("2-digit jersey read as a SINGLE digit")
print(f"  total                {len(short):>4}  of {len(two)} two-digit reads")
print(f"  trailing digit       {len(trailing):>4}   <- what folding assumes")
print(f"  leading digit        {len(leading):>4}   <- 'cannot happen'")
print(f"  neither digit        {len(neither):>4}")
if leading:
    print("    leading-digit cases:", Counter(f"{t}->{p}" for t, p in leading).most_common())
if neither:
    print("    neither:", Counter(f"{t}->{p}" for t, p in neither).most_common(8))

# (c): capture risk -- a genuine 1-digit jersey misread as a 2-digit ending in it
print("\n1-digit jersey read as TWO digits")
grew = [(t, p) for t, p in one if len(p) == 2]
ends_with = [(t, p) for t, p in grew if p[1] == t]
print(f"  total                {len(grew):>4}  of {len(one)} one-digit reads")
print(f"  ends with the truth  {len(ends_with):>4}   <- folding would capture the real digit")
if ends_with:
    print("    cases:", Counter(f"{t}->{p}" for t, p in ends_with).most_common())

print("\n2-digit jersey, full correct reads:",
      sum(1 for t, p in two if t == p), "of", len(two))
print("1-digit jersey, full correct reads:",
      sum(1 for t, p in one if t == p), "of", len(one))

print("\nsensitivity to the confidence gate")
print(f"{'gate':>6} {'reads':>6} {'2d->1d':>7} {'trailing':>9} {'leading':>8}")
for gate in (0.0, 0.25, 0.5, 0.75, 0.9):
    g = {int(i): str(r["text"]).strip() for i, r in raw.items()
         if r["confidence"] >= gate and str(r["text"]).strip().isdigit()
         and 1 <= len(str(r["text"]).strip()) <= 2}
    rows = [(truth[i], g[i]) for i in truth if i in g]
    sh = [(t, p) for t, p in rows if len(t) == 2 and len(p) == 1]
    print(f"{gate:>6.2f} {len(rows):>6} {len(sh):>7} "
          f"{sum(1 for t, p in sh if p == t[1]):>9} "
          f"{sum(1 for t, p in sh if p == t[0] and p != t[1]):>8}")
