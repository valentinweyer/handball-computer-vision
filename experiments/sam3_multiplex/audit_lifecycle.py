"""Exercise late insertion, correction, removal and reset with real GPU inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from experiments.sam3_multiplex.adapter import ROOT, MultiplexAdapter, build_tracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / "sam3"))
    model, loading = build_tracker()
    predictor = MultiplexAdapter(model)
    with np.load(ROOT / "outputs/team_comparison/.FelixClaar_detections_v1.npz") as cache:
        boxes = cache["boxes"][:3].copy()
    checks, outputs = [], []
    with tempfile.TemporaryDirectory(prefix="sam31-lifecycle-") as tmp:
        for f in sorted((ROOT / "data/cache/frames/FelixClaar").glob("*.jpg"))[:8]:
            (Path(tmp) / f.name).symlink_to(f.resolve())
        state = predictor.init_state(tmp)

        def prompt(oid, fid, box):
            predictor.add_new_points_or_box(state, fid, oid, box)

        def group(oid):
            return predictor._group_of(state, oid)

        def snapshot(oid, fid):
            out = group(oid)["output_dict_per_obj"][group(oid)["obj_id_to_idx"][oid]]
            row = out["cond_frame_outputs"].get(fid)
            if row is None:
                row = out["non_cond_frame_outputs"][fid]
            return row["pred_masks"].clone()

        def retained(oid, fid, saved):
            assert torch.equal(snapshot(oid, fid), saved), (oid, fid, "past prediction changed")

        def propagate(start, end, expected, buckets=1):
            seen = []
            for fid, ids, logits in predictor.propagate_in_video(state, start, end-start):
                assert set(ids) == set(expected), (ids, expected)
                assert logits.is_cuda and torch.isfinite(logits).all()
                assert len(logits) == len(expected)
                assert predictor.bucket_count(state) == buckets
                seen.append(fid)
                outputs.append({"frame": fid, "ids": list(map(int, ids)),
                                "nonempty_masks": int((logits > 0).flatten(1).any(1).sum()),
                                "buckets": predictor.bucket_count(state),
                                "groups": len(state["groups"])})
            assert seen == list(range(start, end+1))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prompt(101, 0, boxes[0]); prompt(202, 0, boxes[1])
            propagate(0, 2, [101, 202])
            saved = {oid: snapshot(oid, 1) for oid in [101, 202]}
            prompt(303, 2, boxes[2])
            propagate(2, 3, [101, 202, 303])
            for oid in saved: retained(oid, 1, saved[oid])
            assert 2 in group(303)["output_dict"]["cond_frame_outputs"]
            checks.append("late addition preserves both prior players and historical predictions")

            saved = {oid: snapshot(oid, 2) for oid in [101, 303]}
            prompt(202, 3, boxes[1])
            propagate(3, 4, [101, 202, 303])
            for oid in saved: retained(oid, 2, saved[oid])
            checks.append("reprompt preserves other players' historical predictions")

            saved = {oid: snapshot(oid, 3) for oid in [101, 303]}
            predictor.remove_object(state, 202)
            assert state["obj_id_to_idx"] == {101: 0, 303: 1}
            for oid in saved: retained(oid, 3, saved[oid])
            checks.append("middle removal remaps IDs without losing survivor history")

            saved = snapshot(303, 3)
            predictor.remove_object(state, 101)
            prompt(101, 4, boxes[0])
            propagate(4, 5, [303, 101])
            retained(303, 3, saved)
            fresh = group(101)["output_dict_per_obj"][group(101)["obj_id_to_idx"][101]]
            assert 1 not in fresh["cond_frame_outputs"] and 1 not in fresh["non_cond_frame_outputs"]
            checks.append("remove/re-add resets one identity and retains the survivor's history")

            # The BHC-FAG crash: more late players than the first group has
            # never-used slots. They must open a second group, and both groups
            # must keep tracking with their histories intact.
            # Added at the last tracked frame, as drive_sam2's checkpoints do.
            saved = {oid: snapshot(oid, 4) for oid in [303, 101]}
            late = list(range(400, 416))
            for i, oid in enumerate(late):
                prompt(oid, 5, boxes[i % len(boxes)])
            propagate(5, 7, [303, 101] + late, buckets=2)
            for oid in saved: retained(oid, 4, saved[oid])
            assert len(state["groups"]) == 2
            second = [oid for oid in late if group(oid) is not group(303)]
            assert second and all(4 not in group(oid)["output_dict"]["non_cond_frame_outputs"]
                                  and 4 not in group(oid)["output_dict"]["cond_frame_outputs"]
                                  for oid in second)
            checks.append("overflowing the first group opens a second one with no history; no crash")
    report = {"passed": True, "strict_loading": loading, "checks": checks,
              "frames": outputs, "scope": "API/state integrity on a short real-video GPU probe; not handball identity-quality equivalence"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
