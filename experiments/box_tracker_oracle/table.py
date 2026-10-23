"""Markdown table of every run in the study: correct / wrong-identity / fragments."""
from __future__ import annotations

import json

from experiments.box_tracker_oracle.common import OUT

TAGS = ("baseline", "sort_strict", "colour_q95", "colour_q99", "colour_q95_strict",
        "prtreid_q95", "filled")


def main():
    rows = {}
    for tag in TAGS:
        path = OUT / f"{tag}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        suffix = {"colour_q99": " (q99)"}.get(tag, "")
        for key, m in data["results"].items():
            clip, arm, name = key.split("/")
            rows.setdefault((arm, name + suffix), {})[clip] = m
    print("| arm | variant | FelixClaar | Han-Ber4 | BHC-FAG |")
    print("|---|---|---|---|---|")
    order = {"oracle": 0, "filled": 1, "detector": 2}
    for (arm, name), clips in sorted(rows.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])):
        cells = []
        for clip in ("felix", "han", "bhc"):
            m = clips.get(clip)
            cells.append("" if m is None else
                         f"{m['correct_pct']:.1f} / {m['wrong_identity_pct']:.1f} / "
                         f"{m['mean_fragments_per_reference_id']:.2f}")
        print(f"| {arm} | {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
