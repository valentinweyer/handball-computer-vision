"""Queued, frozen ARC review experiment. No reference identities are assumed."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback


def write(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gate(protocol):
    """Wait for the two authorized dependencies, including their bundle stage."""
    reasons = []
    processes = subprocess.check_output(["ps", "-eo", "comm=,args="], text=True).splitlines()
    for line in processes:
        command, _, arguments = line.strip().partition(" ")
        if command.startswith("python") and any("-m " + name + " " in arguments for name in
               ("scripts.process_video", "scripts.render_full_pipeline")):
            reasons.append("inference process still active: " + line[:220])
    for work in protocol["wait_for"]:
        p = Path(work)
        timings = json.loads((p / "timings.json").read_text()) if (p / "timings.json").exists() else {}
        if not all(k in timings for k in ("track", "bundle")):
            reasons.append("not completed: " + p.name)
        if not all((p / name).is_file() for name in
                   ("track.json", "track_geometry.npz", "track_reads.json")):
            reasons.append("missing completed tracking outputs: " + p.name)
    return reasons


def check_inputs(p):
    for item in p["inputs"].values():
        assert sha(item["path"]) == item["sha256"], "input changed: " + item["path"]


def detect(root, p):
    import numpy as np
    import torch
    import cv2
    from handball_cv.detection.rfdetr_torchscript import TorchScriptDetector
    from handball_cv.pipeline.scan import scan_video
    from handball_cv.video import VideoFrames
    dest = root / "detections.npz"
    if dest.exists():
        with np.load(dest) as d:
            assert int(d["stride"]) == 1 and int(d["scored_frames"]) == p["frames"]
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    model = TorchScriptDetector(Path(p["inputs"]["detector"]["path"]), device="cuda")
    out = scan_video(VideoFrames(Path(p["inputs"]["video"]["path"]), count=p["frames"]),
                     detector=model, motion=False, detect_stride=1, threshold=0.3,
                     detector_id=p["inputs"]["detector"]["path"], desc="every-frame detection")
    with (root / "detections.tmp").open("wb") as f:
        np.savez_compressed(f, **out["detections"])
    (root / "detections.tmp").replace(dest)
    write(root / "detector-complete.json", {"stride": 1, "scored_frames": p["frames"],
                                           "sha256": sha(dest), "device": "cuda"})


def track(root, p):
    import cv2
    import numpy as np
    import torch
    from types import SimpleNamespace
    from arcsort.tracker import load_config
    from experiments.arc_sort_cpu.run import ArcAdapter, encode, frames, prepare_model
    from experiments.arc_team_repair.repair import repair_tracks, RepairConfig
    from experiments.arc_team_repair.run import READ_POLICY
    from scripts.evaluate_tracker_identity import iou_matrix
    from scripts.render_raw_team_classification import load_detection_cache, person_detections
    from handball_cv.teams.model import TeamModel, crop_quality, jersey_color_features, torso_contamination
    if (root / "tracking-complete.json").exists():
        return
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    cache = load_detection_cache(root / "detections.npz")
    assert len(cache["offsets"]) == p["frames"] + 1
    assert READ_POLICY == p["read_policy"]
    assert vars(RepairConfig()) == p["repair_config"]
    model = TeamModel.load(Path(p["inputs"]["team"]["path"]), classifier=SimpleNamespace(), device="cpu")
    net = prepare_model(Path(p["vendor"]), Path(p["inputs"]["osnet"]["path"]))
    cfg = load_config(Path(p["vendor"]) / "arcsort/configs/sportsmot.yaml")
    cfg.update(track_high_thresh=0.5, new_track_thresh=0.5)
    assert cfg == p["arc_config"]
    tracker = ArcAdapter(cfg, {})
    reads = []
    n = 0
    for f, bgr in enumerate(frames(Path(p["inputs"]["video"]["path"]))):
        if f >= p["frames"]:
            break
        det = person_detections(cache, f)
        emb = encode(net, bgr, det.xyxy)
        # Exactly the existing CPU-cache interface, bounded to one frame.
        tracker.tracker.embedding_computer._cache = {f + 1: (det.xyxy.astype(np.float64), emb)}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tracked = tracker.update(det, rgb)
        boxes = det.xyxy
        if len(boxes) and len(tracked):
            crops = model._crop_boxes(rgb, boxes)
            teams, confs = model._color_prediction(jersey_color_features(crops))
            quality = [crop_quality(c) for c in crops]
            contam = torso_contamination(boxes)
            overlap = iou_matrix(boxes, boxes)
            np.fill_diagonal(overlap, 0.)
            pairs = iou_matrix(tracked.xyxy, boxes)
            for row, tid in enumerate(tracked.tracker_id):
                col = int(pairs[row].argmax())
                rival = max(np.delete(pairs[row], col).max(initial=0),
                            np.delete(pairs[:, col], row).max(initial=0))
                if (int(det.class_id[col]) == 2 and confs[col] >= READ_POLICY["confidence"]
                        and quality[col].accepted and quality[col].score >= READ_POLICY["quality"]
                        and contam[col] <= READ_POLICY["contamination_max"]
                        and overlap[col].max() < READ_POLICY["other_box_iou_max"]
                        and pairs[row, col] >= READ_POLICY["match_iou_min"]
                        and pairs[row, col] - rival >= READ_POLICY["match_margin_min"]):
                    reads.append({"frame": f, "track": int(tid), "team": int(teams[col]),
                                  "confidence": float(confs[col]), "quality": quality[col].score,
                                  "detector_index": col})
        n += 1
        if n % 25 == 0:
            write(root / "progress.json", {"phase": "tracking", "frames": n, "total": p["frames"]})
            print("CPU ARC", n, "/", p["frames"], flush=True)
    assert n == p["frames"]
    original = tracker.rows
    observations = {(r["frame"], r["track"]): r["team"] for r in reads}
    repaired, audit = repair_tracks(original, observations, enabled=True, config=RepairConfig())
    disabled, _ = repair_tracks(original, observations, enabled=False)
    assert disabled == original
    assert all(a.split(",")[:1] + a.split(",")[2:] == b.split(",")[:1] + b.split(",")[2:]
               for a, b in zip(original, repaired))
    for name, rows in [("baseline", original), ("repaired", repaired)]:
        (root / (name + ".txt")).write_text("\n".join(rows) + "\n")
    write(root / "team_reads.json", {"policy": READ_POLICY, "reads": reads})
    write(root / "repair_audit.json", audit)
    write(root / "native_arbitration.json", tracker.tracker.arbitration_swaps)
    assert not torch.cuda.is_initialized()
    write(root / "tracking-complete.json", {"frames": n, "cuda_initialized": False,
                                            "corrections": len(audit["corrections"]),
                                            "changed_rows": audit["changed_rows"],
                                            "accuracy": None, "ground_truth": False})


def selection(original, repaired, audit, total, fps, seed=20261022, controls=12):
    rng = random.Random(seed)
    changed = []
    for a, b in zip(original, repaired):
        if a != b:
            v = a.split(",")
            changed.append((int(v[0]) - 1, int(v[1])))
    items = []
    def interval(e, correction):
        start = max(0, e["start"] - int(3 * fps))
        end = min(total - 1, e["end"] + int(4 * fps))
        if correction:
            anchor = e["post"]
            confirmed = max(a["last"] for a in anchor) if isinstance(anchor, list) else anchor["last"]
            end = min(total - 1, max(end, confirmed + int(3 * fps)))
        else:
            start = max(start, end - int(12 * fps) + 1)
        return start, end
    for e in audit["corrections"]:
        start, end = interval(e, True)
        items.append({"kind": "correction", "start": start, "end": end,
                      "focus": e["pair"], "event": e})
    pool = []
    for e in audit["encounters"]:
        start, end = interval(e, False)
        if e.get("decision") == "repair" or any(start <= f <= end and t in e["pair"] for f, t in changed):
            continue
        pool.append({"kind": "unchanged_control", "start": start, "end": end,
                     "focus": e["pair"], "event": e})
    rng.shuffle(pool)
    chosen = []
    for item in pool:
        if not any(max(item["start"], other["start"]) <= min(item["end"], other["end"]) for other in chosen):
            chosen.append(item)
        if len(chosen) == controls:
            break
    items.extend(chosen)
    rng.shuffle(items)
    for i, item in enumerate(items, 1):
        item["id"] = f"case-{i:03d}"
        item["video"] = item["id"] + ".mp4"
    return items


def render(root, p):
    import cv2
    import numpy as np
    from experiments.arc_team_repair.repair import parse_lines
    from handball_cv.video import ffmpeg_exe
    cv2.setNumThreads(1)
    original = (root / "baseline.txt").read_text().splitlines()
    repaired = (root / "repaired.txt").read_text().splitlines()
    audit = json.loads((root / "repair_audit.json").read_text())
    items = selection(original, repaired, audit, p["frames"], p["fps"], p["sample_seed"], p["controls"])
    review = root / "review"
    review.mkdir(exist_ok=True)
    paths = [parse_lines(original), parse_lines(repaired)]
    cap = cv2.VideoCapture(p["inputs"]["video"]["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    width = min(960, p["width"])
    height = int(round(p["height"] * width / p["width"] / 2)) * 2
    scale = width / p["width"]
    try:
        for item in items:
            dest = review / item["video"]
            if dest.exists():
                continue
            temp = dest.with_name(dest.stem + ".partial.mp4")
            command = [ffmpeg_exe(), "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo",
                       "-pix_fmt", "bgr24", "-s", f"{2*width}x{height+40}", "-r", str(p["fps"]),
                       "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                       "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(temp)]
            cap.set(cv2.CAP_PROP_POS_FRAMES, item["start"])
            with (root / "render-errors.log").open("ab") as log:
                proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                try:
                    for f in range(item["start"], item["end"] + 1):
                        ok, bgr = cap.read()
                        if not ok:
                            raise RuntimeError(f"decode failed at {f}")
                        canvas = np.zeros((height + 40, width * 2, 3), np.uint8)
                        for side in (0, 1):
                            pane = cv2.resize(bgr, (width, height))
                            for tid, box in paths[side].get(f, {}).items():
                                xy = np.rint(box * scale).astype(int)
                                focused = tid in item["focus"]
                                colour = ((37*tid % 180 + 60), (97*tid % 180 + 60), (61*tid % 180 + 60)) if focused else (130,130,130)
                                cv2.rectangle(pane, tuple(xy[:2]), tuple(xy[2:]), colour, 3 if focused else 1)
                                if focused:
                                    cv2.putText(pane, f"T{tid}", (xy[0], max(22, xy[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 5)
                                    cv2.putText(pane, f"T{tid}", (xy[0], max(22, xy[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, colour, 2)
                            canvas[40:, side*width:(side+1)*width] = pane
                            title = ("A: Original ARC" if side == 0 else "B: Team repair") + f" | {f/p['fps']:.2f}s"
                            cv2.putText(canvas, title, (side*width+12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
                        proc.stdin.write(canvas.tobytes())
                finally:
                    proc.stdin.close()
                    result = proc.wait()
                if result:
                    raise RuntimeError("video encoder failed; see render-errors.log")
            temp.replace(dest)
            print("rendered", item["id"], flush=True)
    finally:
        cap.release()
    write(review / "manifest.json", {"seed": p["sample_seed"], "items": items,
                                      "ground_truth": False, "scope": "event review, not overall accuracy"})
    public = [{k: item[k] for k in ("id", "video", "start", "end", "focus")} for item in items]
    template = (root / "review-template.html").read_text()
    (review / "index.html").write_text(template.replace("__ITEMS__", json.dumps(public)).replace("__FPS__", str(p["fps"])))
    write(root / "complete.json", {"complete": True, "review_cases": len(items),
                                   "corrections": sum(i["kind"] == "correction" for i in items),
                                   "controls": sum(i["kind"] == "unchanged_control" for i in items),
                                   "ground_truth": False, "accuracy": None})


def workflow(root, p):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    env["PYTHONPATH"] = ":".join([p["deps"], p["vendor"], str(root / "frozen/src"), str(root / "frozen")])
    for stage in ("detect", "track", "render"):
        check_inputs(p)
        write(root / "status.json", {"stage": stage, "updated_at": time.time(), "pid": os.getpid()})
        child_env = dict(env)
        if stage != "detect":
            child_env["CUDA_VISIBLE_DEVICES"] = ""
        with (root / (stage + ".log")).open("a") as log:
            subprocess.run([sys.executable, str(root / "runner.py"), "--root", str(root), "--worker", stage],
                           cwd=root / "frozen", env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    check_inputs(p)
    write(root / "status.json", {"stage": "complete", "updated_at": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", choices=["workflow", "detect", "track", "render"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    p = json.loads((root / "protocol.json").read_text())
    assert sha(root / "runner.py") == p["runner_sha256"]
    if args.worker:
        try:
            {"workflow": workflow, "detect": detect, "track": track, "render": render}[args.worker](root, p)
        except Exception as exc:
            previous = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
            if args.worker == "workflow" and previous.get("stage") == "failed":
                previous["workflow_error"] = str(exc)
                write(root / "status.json", previous)
            else:
                write(root / "status.json", {"stage": "failed", "worker": args.worker,
                                             "error": str(exc), "traceback": traceback.format_exc(), "updated_at": time.time()})
            raise
        return
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "launched.json").exists():
            print((root / "launched.json").read_text())
            return
        reasons = gate(p)
        if reasons or args.check:
            print(json.dumps({"ready": not reasons, "waiting": reasons}))
            return
        check_inputs(p)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
        with (root / "workflow.log").open("a") as log:
            proc = subprocess.Popen(["taskset", "-c", "0,1", "nice", "-n", "15", "ionice", "-c", "3",
                                     sys.executable, str(root / "runner.py"), "--root", str(root), "--worker", "workflow"],
                                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        write(root / "launched.json", {"pid": proc.pid, "started_at": time.time(), "protocol_sha256": sha(root / "protocol.json")})
        print((root / "launched.json").read_text())


if __name__ == "__main__":
    main()
