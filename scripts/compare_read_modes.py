"""Compare whole-number reading against digit-wise reading on the same crops.

Both arms come from the same model on the same images, so this is a *paired*
comparison and is judged as one: McNemar's exact test on the crops where the two
modes disagree, not a difference of two accuracy figures. With a few hundred
samples a three-point accuracy gap is noise, and this project has already been
bitten once by reading a threshold off a single clip.

Two questions are asked separately, because they can move in opposite directions:

  H1 (accuracy)  does digit-wise reading get more numbers right?
  H2 (honesty)   does it convert *silent truncations* into *explicit partials*?
                 Whole-number mode answers a half-visible 17 with a confident
                 "7", which the voter counts as a wrong vote. Digit mode can say
                 "? 7", which is evidence the voter can fold instead. H2 may
                 matter more than H1: it is the failure that forced suffix-folding
                 into `NumberVoter` in the first place.

Usage:
    python -m scripts.compare_read_modes runs/number_eval_1080p/qwen38_flash_next_benchmark.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import comb
from pathlib import Path

from scripts.label_jersey_numbers import normalize_prediction


# Defaults name the reasoning-enabled pair. The same analysis runs on the
# reasoning-disabled pair via --whole/--digit: reasoning was measured to
# dominate abstention regardless of read mode, so digit-vs-whole has to be
# answered at both settings, not just the one that happened to run first.
WHOLE_VARIANT = "context"
DIGIT_VARIANT = "context_digits"


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts.

    Under the null the discordant pairs split 50/50, so this is a two-sided
    binomial test on `only_a` successes out of `only_a + only_b` trials.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(only_a, only_b) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def load_truth(labels_path: Path) -> tuple[dict, set, set]:
    doc = json.loads(labels_path.read_text())
    readable, unreadable, unsure = {}, set(), set()
    for index, label in doc.get("labels", {}).items():
        status = label.get("status")
        if status == "readable" and label.get("value"):
            readable[int(index)] = str(label["value"])
        elif status == "unreadable":
            unreadable.add(int(index))
        elif status == "unsure":
            unsure.add(int(index))
    return readable, unreadable, unsure


def pair_variants(report: dict, indices, whole=WHOLE_VARIANT, digit=DIGIT_VARIANT) -> dict:
    """-> {index: {variant: (prediction, parse_status)}} for indices both modes have."""
    raw = report.get("raw", {})
    whole_raw, digit_raw = raw.get(whole, {}), raw.get(digit, {})
    paired = {}
    for index in indices:
        key = str(index)
        if key not in whole_raw or key not in digit_raw:
            continue
        # A truncated response is a harness failure, not a model answer; excluding
        # it keeps a token-budget overrun from being scored as an abstention.
        if (whole_raw[key]["parse_status"] == "truncated"
                or digit_raw[key]["parse_status"] == "truncated"):
            continue
        paired[index] = {
            whole: (normalize_prediction(whole_raw[key]["prediction"]),
                    whole_raw[key]["parse_status"]),
            digit: (normalize_prediction(digit_raw[key]["prediction"]),
                    digit_raw[key]["parse_status"]),
        }
    return paired


def score_arm(paired: dict, variant: str, readable: dict, unreadable: set) -> dict:
    answered = correct = wrong = 0
    abstained_on_readable = 0
    abstained_on_unreadable = 0
    answered_on_unreadable = 0
    for index, arms in paired.items():
        prediction, _status = arms[variant]
        if index in readable:
            if prediction:
                answered += 1
                if prediction == readable[index]:
                    correct += 1
                else:
                    wrong += 1
            else:
                abstained_on_readable += 1
        elif index in unreadable:
            if prediction:
                answered_on_unreadable += 1
            else:
                abstained_on_unreadable += 1
    n_readable = correct + wrong + abstained_on_readable
    n_unreadable = answered_on_unreadable + abstained_on_unreadable
    return {
        "readable_crops": n_readable,
        "coverage": answered / n_readable if n_readable else 0.0,
        "accuracy": correct / n_readable if n_readable else 0.0,
        "selective_accuracy": correct / answered if answered else 0.0,
        "correct": correct,
        "wrong": wrong,
        "unreadable_crops": n_unreadable,
        "abstention_rate": (
            abstained_on_unreadable / n_unreadable if n_unreadable else 0.0
        ),
        "false_reads_on_unreadable": answered_on_unreadable,
    }


def compare_h1(paired: dict, readable: dict, whole=WHOLE_VARIANT, digit=DIGIT_VARIANT) -> dict:
    """Paired accuracy comparison on crops with a known number."""
    both = whole_only = digit_only = neither = 0
    for index, arms in paired.items():
        if index not in readable:
            continue
        truth = readable[index]
        w = arms[whole][0] == truth
        d = arms[digit][0] == truth
        both += w and d
        whole_only += w and not d
        digit_only += d and not w
        neither += not w and not d
    return {
        "both_correct": both,
        "whole_only": whole_only,
        "digit_only": digit_only,
        "neither": neither,
        "p_value": mcnemar_exact(whole_only, digit_only),
    }


def classify_truncations(paired: dict, readable: dict, whole=WHOLE_VARIANT,
                         digit=DIGIT_VARIANT) -> dict:
    """H2: on two-digit numbers, what does each mode do when it cannot read it all?

    A *silent truncation* is whole-number mode confidently returning one digit of a
    two-digit number. The digit-mode counterpart is an explicit partial ("? 7"),
    which carries the same information without asserting a wrong number.
    """
    silent = []
    for index, arms in paired.items():
        truth = readable.get(index)
        if not truth or len(truth) != 2:
            continue
        w_pred, _ = arms[whole]
        d_pred, d_status = arms[digit]
        if len(w_pred) == 1 and w_pred in truth:
            silent.append((index, truth, w_pred, d_pred, d_status))
    rescued = [row for row in silent if row[4].startswith("partial_")]
    solved = [row for row in silent if row[3] == row[1]]
    return {
        "silent_truncations_in_whole_mode": len(silent),
        "digit_mode_returned_explicit_partial": len(rescued),
        "digit_mode_read_the_whole_number": len(solved),
        "examples": [
            {"truth": t, "whole": w, "digit": d, "digit_status": s}
            for _i, t, w, d, s in silent[:12]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--whole", default=WHOLE_VARIANT)
    parser.add_argument("--digit", default=DIGIT_VARIANT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    readable, unreadable, unsure = load_truth(args.labels)
    paired = pair_variants(report, sorted(set(readable) | unreadable),
                           args.whole, args.digit)

    result = {
        "report": str(args.report),
        "labels": str(args.labels),
        "paired_crops": len(paired),
        "unsure_excluded": len(unsure),
        "arms": {
            args.whole: score_arm(paired, args.whole, readable, unreadable),
            args.digit: score_arm(paired, args.digit, readable, unreadable),
        },
        "h1_accuracy": compare_h1(paired, readable, args.whole, args.digit),
        "h2_partial_honesty": classify_truncations(paired, readable,
                                                   args.whole, args.digit),
    }

    if args.dataset:
        ds = json.loads(args.dataset.read_text())
        by_index = {x["index"]: x for x in ds["samples"]}
        for key, field in (("by_clip", "source_clip"), ("by_band", "height_band")):
            groups = defaultdict(list)
            for index in paired:
                if index in by_index:
                    groups[by_index[index][field]].append(index)
            result[key] = {
                name: {
                    v: score_arm({i: paired[i] for i in idxs}, v, readable, unreadable)
                    for v in (args.whole, args.digit)
                }
                for name, idxs in sorted(groups.items())
            }

    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
