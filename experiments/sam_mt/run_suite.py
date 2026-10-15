"""Run native SAM-MT/SAM2 paired clips sequentially, with fresh warmed processes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for repeat in range(1, args.repeats + 1):
        for index, case in enumerate(("felix", "han", "bhc")):
            backends = ("sam_mt", "sam2") if (repeat + index) % 2 else ("sam2", "sam_mt")
            for backend in backends:
                name = f"{case}_{backend}_r{repeat}"
                target = args.output / name
                print(f"START {name}", flush=True)
                with (args.output / f"{name}.log").open("w") as log:
                    subprocess.run([
                        sys.executable, "-m", "experiments.sam_mt.benchmark",
                        "--backend", backend, "--case", case, "--output", str(target),
                    ], stdout=log, stderr=subprocess.STDOUT, check=True)
                with (args.output / f"{name}_score.log").open("w") as log:
                    subprocess.run([
                        sys.executable, "-m", "experiments.efficienttam.score_comparison",
                        "--run", str(target),
                    ], stdout=log, stderr=subprocess.STDOUT, check=True)
                manifest = json.loads((target / "manifest.json").read_text())
                metrics = json.loads((target / "score.json").read_text())["metrics"]
                row = {"run": str(target), "case": case, "backend": backend, "repeat": repeat,
                       "fps": manifest["loop_fps"], "peak_cuda_bytes": manifest["cuda_peak_allocated_bytes"],
                       **metrics}
                results.append(row)
                (args.output / "summary.json").write_text(json.dumps(results, indent=2))
                print(f"DONE {name}: {row['fps']:.2f} FPS, {row['correct_pct']:.2f}% correct, "
                      f"{row['mixed_tracklets']} mixed, {row['id_switches']} switches", flush=True)


if __name__ == "__main__":
    main()
