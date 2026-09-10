"""EfficientTAM lifecycle smoke test; NOT a quality/speed benchmark.

Run in a fresh process with a pinned EfficientTAM checkout on PYTHONPATH:
    PYTHONPATH=/path/to/EfficientTAM uv run --no-sync python -m \
        experiments.efficienttam.audit_lifecycle --frames data/cache/frames/Han-Ber4_cached

Uses real neural inference, eight existing JPEGs and three synthetic boxes.
Weights are random unless --checkpoint selects an existing trained Small file.
No packages/weights are installed and no input files are edited.
Missing optional hole-filling extension warnings are retained in the report.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np
import torch


def compare_lifecycle(sam2_file: Path, efficienttam_file: Path) -> dict[str, bool]:
    """Compare executable ASTs, normalizing only the model name in strings."""
    def methods(path):
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        result = {}
        for node in cls.body:
            if isinstance(node, ast.FunctionDef):
                for part in ast.walk(node):
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        part.value = part.value.replace("EfficientTAM", "SAM2")
                result[node.name] = ast.dump(node, include_attributes=False)
        return result

    baseline, candidate = methods(sam2_file), methods(efficienttam_file)
    names = (
        "init_state", "_obj_id_to_idx", "add_new_points_or_box", "add_new_mask",
        "propagate_in_video_preflight", "propagate_in_video", "remove_object",
        "reset_state", "clear_all_prompts_in_frame", "_get_orig_video_res_output",
    )
    return {name: baseline[name] == candidate[name] for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional trained Small weights; default is the weight-free API probe")
    parser.add_argument("--sam2-source", type=Path,
                        default=Path("sam2-upstream/sam2/sam2_video_predictor.py"))
    args = parser.parse_args()
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("checkpoint must be an existing file")
    import efficient_track_anything
    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
    from handball_cv.tracking.sam2_driver import masks_from_logits
    import supervision as sv

    candidate = Path(efficient_track_anything.__file__).parent / "efficienttam_video_predictor.py"
    equality = compare_lifecycle(args.sam2_source, candidate)
    assert all(equality.values()), equality
    inputs = sorted(args.frames.glob("*.jpg"), key=lambda p: int(p.stem))[:8]
    assert len(inputs) == 8, "need eight JPEG frames"
    torch.manual_seed(0)
    checks = []
    with warnings.catch_warnings(record=True) as recorded, tempfile.TemporaryDirectory() as tmp:
        warnings.simplefilter("always")
        for i, path in enumerate(inputs):
            shutil.copyfile(path, Path(tmp) / f"{i:05d}.jpg")
        predictor = build_efficienttam_video_predictor(
            "configs/efficienttam/efficienttam_s.yaml",
            ckpt_path=str(args.checkpoint) if args.checkpoint else None,
            vos_optimized=False,
            hydra_overrides_extra=["++model.compile_image_encoder=false"],
        )
        assert predictor.image_size == 1024 and predictor.num_maskmem == 7
        state = predictor.init_state(video_path=tmp)
        height, width = state["video_height"], state["video_width"]
        boxes = {
            11: np.array([.1, .2, .2, .7]) * [width, height, width, height],
            22: np.array([.4, .2, .5, .7]) * [width, height, width, height],
            33: np.array([.7, .2, .8, .7]) * [width, height, width, height],
        }

        def prompt(oid, frame):
            fid, ids, logits = predictor.add_new_points_or_box(
                state, frame_idx=frame, obj_id=oid,
                box=boxes[oid].astype(np.float32), clear_old_points=True,
            )
            assert fid == frame and ids == state["obj_ids"]
            assert logits.shape == (len(ids), 1, height, width)

        def snapshot(oid):
            idx = state["obj_id_to_idx"][oid]
            stores = {key: state[key][idx] for key in (
                "point_inputs_per_obj", "mask_inputs_per_obj", "output_dict_per_obj",
                "temp_output_dict_per_obj", "frames_tracked_per_obj",
            )}
            # Check both container retention and actual memory values, by public ID.
            outputs = {key: dict(group) for key, group in stores["output_dict_per_obj"].items()}
            tensors = [(out, out["maskmem_features"], out["maskmem_features"].clone())
                       for group in outputs.values()
                       for out in group.values() if out["maskmem_features"] is not None]
            return stores, outputs, tensors

        def retained(oid, saved):
            idx = state["obj_id_to_idx"][oid]
            stores, outputs, tensors = saved
            assert all(state[key][idx] is value for key, value in stores.items())
            current = state["output_dict_per_obj"][idx]
            for key, group in outputs.items():
                assert current[key].keys() == group.keys()
                assert all(current[key][fid] is out for fid, out in group.items())
            assert all(out["maskmem_features"] is tensor and torch.equal(tensor, copy)
                       for out, tensor, copy in tensors)

        def propagate(start, end, expected):
            seen = []
            for fid, ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=start, max_frame_num_to_track=end-start,
            ):
                assert ids == expected
                assert logits.shape == (len(ids), 1, height, width)
                assert torch.isfinite(logits).all()
                masks = masks_from_logits(logits)
                assert masks.shape == (len(ids), height, width) and masks.dtype == bool
                assert sv.mask_to_xyxy(masks=masks).shape == (len(ids), 4)
                seen.append(fid)
            assert seen == list(range(start, end+1)), seen

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prompt(11, 0)
            prompt(22, 0)
            propagate(0, 2, [11, 22])
            saved = {oid: snapshot(oid) for oid in (11, 22)}
            prompt(33, 2)
            for oid in saved:
                retained(oid, saved[oid])
            checks.append("late addition preserves existing object memory")
            propagate(2, 3, [11, 22, 33])
            saved = snapshot(22)
            original = state["output_dict_per_obj"][state["obj_id_to_idx"][11]]
            prompt(11, 3)
            retained(22, saved)
            assert state["output_dict_per_obj"][state["obj_id_to_idx"][11]] is original
            checks.append("in-place correction retains history and other objects")
            propagate(3, 4, [11, 22, 33])
            saved = {oid: snapshot(oid) for oid in (11, 33)}
            predictor.remove_object(state, obj_id=22)
            for oid in saved:
                retained(oid, saved[oid])
            assert state["obj_id_to_idx"] == {11: 0, 33: 1}
            checks.append("middle removal remaps indices without losing survivor memory")
            saved = snapshot(33)
            predictor.remove_object(state, obj_id=11)
            prompt(11, 4)
            retained(33, saved)
            idx = state["obj_id_to_idx"][11]
            assert not state["frames_tracked_per_obj"][idx]
            assert all(not v for v in state["output_dict_per_obj"][idx].values())
            assert set(state["temp_output_dict_per_obj"][idx]["cond_frame_outputs"]) == {4}
            checks.append("same-ID reset creates fresh target memory; survivor unchanged")
            propagate(4, 6, [33, 11])
            predictor.remove_object(state, obj_id=33)
            # Preserve drive_sam2's existing last-object reset fallback.
            saved = snapshot(11)
            prompt(11, 6)
            retained(11, saved)
            propagate(6, 7, [11])
            checks.append("single-object in-place fallback resumes")
            predictor.remove_object(state, obj_id=11)
            assert not state["obj_ids"] and not state["output_dict_per_obj"]
            prompt(22, 7)
            propagate(7, 7, [22])
            checks.append("last removal clears session objects; later seeding works")
        warning_text = sorted({str(w.message) for w in recorded})
    print(json.dumps({
        "scope": "API/state smoke only; no quality or speed evidence",
        "trained_weights": args.checkpoint is not None,
        "candidate_source": str(candidate), "lifecycle_ast_equal": equality,
        "config": "configs/efficienttam/efficienttam_s.yaml",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "compile_image_encoder": False, "vos_optimized": False,
        "checks": checks, "warnings": warning_text,
    }, indent=2))


if __name__ == "__main__":
    main()
