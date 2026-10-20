"""Compare two process_video work dirs' track summaries (track.json) side by side.

    python -m experiments.sam2_throughput.compare_runs WORK_A WORK_B
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix[:-1], value
    elif isinstance(value, list):
        yield prefix[:-1] + ".len", len(value)


def main():
    a, b = (json.loads((Path(p) / "track.json").read_text()) for p in sys.argv[1:3])
    fa, fb = dict(flatten(a)), dict(flatten(b))
    for key in sorted(set(fa) | set(fb)):
        if key.startswith(("frame_players", "presence")) and not key.endswith("len"):
            continue
        va, vb = fa.get(key), fb.get(key)
        if va != vb or key.count(".") <= 1:
            print(f"{key:55} {va!s:>12} {vb!s:>12}")
    players = lambda run: {int(k): v for k, v in run.get("players", {}).items()} \
        if isinstance(run.get("players"), dict) else None
    numbers = lambda run: sorted(
        (str(p.get("team")), str(p.get("number"))) for p in
        (run["players"].values() if isinstance(run.get("players"), dict) else run.get("players", []))
        if isinstance(p, dict) and p.get("number"))
    na, nb = numbers(a), numbers(b)
    print("\nresolved (team, number) pairs:", len(na), len(nb))
    print("only A:", sorted(set(na) - set(nb)))
    print("only B:", sorted(set(nb) - set(na)))


if __name__ == "__main__":
    main()
