"""Can a VLM label court keypoints? A hand-off harness for an agent labeller.

``prepare`` writes a self-contained task folder -- frames, a numbered court
template, and ``PROMPT.md``. A labeller answers it with ``predictions.json``:
an agent handed the folder by hand, ``codex`` (GPT-6 Astra through the Codex
CLI on the ChatGPT subscription, no API key), or ``mistral`` (the Mistral chat
API, ``MISTRAL_API_KEY``). Both ask one frame at a time and resume where a run
stopped. ``score`` grades the file against the hand labels of the
court-keypoint export and draws overlays to look at.

    python -m scripts.vlm_court_keypoints prepare --name astra_high
    python -m scripts.vlm_court_keypoints codex --name astra_high --effort high
    python -m scripts.vlm_court_keypoints score --name astra_high

Two things decide the design:

**The template is drawn the way the camera sees the court.** That is
``draw_court``'s own picture -- far sideline (y=2000) on top, KP01 top-left --
which is also how the export was annotated (``upright`` holds on 887 of 892
images). Handing the labeller a template in any other orientation would invite
a coherently mirrored labelling, the failure ``TODO.md`` traces the
middle-court error to. The images that are not upright are mirrored annotation
slips and are left out.

**Self-consistency is not correctness.** A mirrored labelling fits a homography
perfectly. So the headline numbers compare each predicted point with the hand
click for the same landmark, and map the hand-labelled points through the
*predicted* homography to see where the court lands in cm.

A landmark the annotator never labelled is reported but not scored. Projecting
it through the hand labels' homography looked like a fair stand-in, but that is
extrapolation: on the first run two such projections landed outside the frame
where the labeller's point was, on inspection, correct.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from dotenv import load_dotenv
from sports import MeasurementUnit
from sports.handball import CourtConfiguration, League, draw_court

from handball_cv.court.keypoints import KEYPOINT_TO_VERTEX, upright
from scripts.benchmark_qwen_jersey_ocr import extract_content, image_data_url
from scripts.recover_court_keypoint_mapping import DEFAULT_DATASET, load_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs/vlm_court_keypoints"

# The export is broadcast footage stretched to 640x640. Frames are handed over
# at 16:9 so the court is not squashed; labels are normalised, so they rescale
# exactly.
FRAME_W, FRAME_H = 1280, 720
GOAL_LINE_X = (0.0, 4000.0)
CLOSE_PX = 10.0
GROSS_PX = 40.0

TEMPLATE_SCALE = 0.35  # px per cm
TEMPLATE_PADDING = 60

# View-relative description of each template vertex, KP number = index + 1.
# "Far" is court y=2000, "near" (camera side) y=0, "left" x=0, "right" x=4000.
DESCRIPTIONS = (
    "far-left corner (left goal line x far sideline)",
    "left goal line, where the far end of the 6 m goal-area line meets it",
    "left goal, far post (on the goal line)",
    "left goal, near post (on the goal line)",
    "left goal line, where the near end of the 6 m goal-area line meets it",
    "near-left corner (left goal line x near sideline)",
    "left 9 m dashed line meets the far sideline",
    "left 6 m line, far end of its straight middle segment",
    "left 6 m line, near end of its straight middle segment",
    "left 9 m dashed line meets the near sideline",
    "left 6 m line, centre point (opposite the goal centre)",
    "left 7 m mark, far end",
    "left 7 m mark, near end",
    "centre line meets the far sideline",
    "centre circle crosses the centre line, far side",
    "centre circle, left-most point",
    "centre point of the court",
    "centre circle, right-most point",
    "centre circle crosses the centre line, near side",
    "centre line meets the near sideline",
    "right 7 m mark, far end",
    "right 7 m mark, near end",
    "right 6 m line, centre point (opposite the goal centre)",
    "right 9 m dashed line meets the far sideline",
    "right 6 m line, far end of its straight middle segment",
    "right 6 m line, near end of its straight middle segment",
    "right 9 m dashed line meets the near sideline",
    "far-right corner (right goal line x far sideline)",
    "right goal line, where the far end of the 6 m goal-area line meets it",
    "right goal, far post (on the goal line)",
    "right goal, near post (on the goal line)",
    "right goal line, where the near end of the 6 m goal-area line meets it",
    "near-right corner (right goal line x near sideline)",
    "left 9 m dashed line, far end of its straight middle segment",
    "left 9 m dashed line, near end of its straight middle segment",
    "right 9 m dashed line, far end of its straight middle segment",
    "right 9 m dashed line, near end of its straight middle segment",
)

PROMPT = """\
# Label handball court keypoints

You are labelling court landmarks in broadcast frames of indoor handball.

## Files in this folder

- `frames/` holds {count} frames, each {w}x{h} pixels.
- `template.png` shows the court from above with 37 numbered landmarks
  (KP01 to KP37). It is drawn **the way the broadcast camera sees the court**:
  the far sideline is at the top, the near (camera-side) sideline at the bottom,
  the goal on the left of a frame is the left goal of the template.

## What to do

For every frame, find each landmark that lies inside the frame and give its
pixel position.

- Coordinates are pixels in that frame: origin at the top-left corner, x to the
  right, y downward, x in 0..{wmax}, y in 0..{hmax}.
- Place each point on the exact intersection of the painted line centres.
- Include a landmark hidden briefly behind a player only if the lines around it
  fix its position precisely. Otherwise leave it out.
- **A landmark with the wrong KP number is much worse than a missing one.** If
  you are not sure which landmark you are looking at, leave it out.
- When no goal is in frame, decide left and right from the centre line: points
  left of the centre line belong to the left half.

## Rules

- Label by looking at the images. Do **not** write or run code to analyse them
  (no OpenCV, no line detection, no scripts of any kind).
- Do not read files outside this folder.

## Landmarks

| KP | where |
|----|-------|
{table}

## Output

Write `predictions.json` in this folder. One entry per frame (an empty object if
you place nothing), mapping KP number to `[x, y]`:

```json
{{
  "frame_01.jpg": {{"KP17": [640.5, 402.0], "KP14": [633.0, 688.5]}},
  "frame_02.jpg": {{}}
}}
```
"""


def court_vertices() -> np.ndarray:
    config = CourtConfiguration(league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS)
    return np.asarray(config.vertices, dtype=float)


def ground_truth(label: np.ndarray) -> dict[int, np.ndarray]:
    """Hand-labelled pixel position of each visible template vertex."""
    return {
        KEYPOINT_TO_VERTEX[slot]: label[slot, :2] * (FRAME_W, FRAME_H)
        for slot in range(len(label))
        if label[slot, 2] > 0
    }


def camera_view(truth: dict[int, np.ndarray], vertices: np.ndarray) -> bool:
    """True if the hand labels see the court the way the template draws it."""
    ids = list(truth)
    return upright(np.array([truth[i] for i in ids]), vertices[ids])


def middle_court(truth: dict[int, np.ndarray], vertices: np.ndarray) -> bool:
    """No goal-line landmark labelled: the framing the keypoint model fails on."""
    return not any(vertices[v, 0] in GOAL_LINE_X for v in truth)


def render_template(vertices: np.ndarray) -> np.ndarray:
    config = CourtConfiguration(league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS)
    court = draw_court(config, scale=TEMPLATE_SCALE, padding=TEMPLATE_PADDING, line_thickness=2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, (x, y) in enumerate(vertices):
        # draw_court's own mapping: x=0 on the left, y=0 at the bottom.
        centre = (round(x * TEMPLATE_SCALE + TEMPLATE_PADDING),
                  round((config.court_width - y) * TEMPLATE_SCALE + TEMPLATE_PADDING))
        cv2.circle(court, centre, 14, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(court, centre, 14, (255, 255, 255), 1, cv2.LINE_AA)
        text = f"{index + 1:02d}"
        (tw, th), _ = cv2.getTextSize(text, font, 0.45, 1)
        cv2.putText(court, text, (centre[0] - tw // 2, centre[1] + th // 2), font, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    width = court.shape[1]
    for text, y in (("FAR SIDELINE (away from camera)", 30),
                    ("NEAR SIDELINE (camera side)", court.shape[0] - 18)):
        (tw, _), _ = cv2.getTextSize(text, font, 0.7, 2)
        cv2.putText(court, text, ((width - tw) // 2, y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return court


def spaced(items: list, count: int) -> list:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, count).round().astype(int)]


def prepare(args: argparse.Namespace) -> None:
    vertices = court_vertices()
    run_dir = RUNS_DIR / args.name
    task_dir = run_dir / "task"
    if task_dir.exists():
        raise SystemExit(f"{task_dir} exists; pick another --name")

    middle, other, skipped = [], [], []
    for label_path in sorted((args.dataset / args.split / "labels").glob("*.txt")):
        label = load_label(label_path)
        if label is None:
            continue
        truth = ground_truth(label)
        if not camera_view(truth, vertices):
            skipped.append(label_path.stem)
            continue
        (middle if middle_court(truth, vertices) else other).append(label_path)
    if not middle and not other:
        raise SystemExit(f"no labels under {args.dataset / args.split}")
    chosen = sorted(spaced(middle, args.middle) + spaced(other, args.other))

    (task_dir / "frames").mkdir(parents=True)
    key = {"dataset": str(args.dataset), "split": args.split,
           "frame_size": [FRAME_W, FRAME_H], "frames": {}}
    for number, label_path in enumerate(chosen, start=1):
        image_path = label_path.parent.parent / "images" / f"{label_path.stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"cannot read {image_path}")
        frame = f"frame_{number:02d}.jpg"
        cv2.imwrite(str(task_dir / "frames" / frame),
                    cv2.resize(image, (FRAME_W, FRAME_H), interpolation=cv2.INTER_CUBIC),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        key["frames"][frame] = {"image": str(image_path), "label": str(label_path),
                                "middle_court": label_path in middle}

    cv2.imwrite(str(task_dir / "template.png"), render_template(vertices))
    table = "\n".join(f"| KP{i + 1:02d} | {text} |" for i, text in enumerate(DESCRIPTIONS))
    (task_dir / "PROMPT.md").write_text(PROMPT.format(
        count=len(chosen), w=FRAME_W, h=FRAME_H, wmax=FRAME_W - 1, hmax=FRAME_H - 1, table=table))
    # Outside task/ so the labeller is not handed the answer key.
    (run_dir / "key.json").write_text(json.dumps(key, indent=2))

    n_middle = sum(entry["middle_court"] for entry in key["frames"].values())
    print(f"{len(chosen)} frames ({n_middle} middle-court) -> {task_dir}")
    if skipped:
        print(f"skipped {len(skipped)} image(s) labelled against the export's orientation: {skipped}")
    print(f"\nRun the labeller in {task_dir} and ask it to follow PROMPT.md; then\n"
          f"  python -m scripts.vlm_court_keypoints score --name {args.name}")


def parse_predictions(raw: dict) -> dict[str, dict[int, np.ndarray]]:
    """``{frame: {"KP17": [x, y]}}`` -> ``{frame: {vertex_index: xy}}``.

    Accepts ``"KP17"``, ``"17"`` or ``"KP17-Center"`` keys and ``[x, y]`` or
    ``{"x": .., "y": ..}`` values. Anything else is an error, not a skip: a
    silently dropped point would read as an abstention.
    """
    parsed: dict[str, dict[int, np.ndarray]] = {}
    for frame, points in raw.items():
        if not isinstance(points, dict):
            raise ValueError(f"{frame}: expected an object of KP -> [x, y]")
        out: dict[int, np.ndarray] = {}
        for key, value in points.items():
            match = re.fullmatch(r"(?:KP)?0*(\d{1,2})(?:-.*)?", str(key).strip(), re.IGNORECASE)
            if not match or not 1 <= int(match.group(1)) <= len(DESCRIPTIONS):
                raise ValueError(f"{frame}: unknown landmark {key!r}")
            if isinstance(value, dict):
                value = [value.get("x"), value.get("y")]
            xy = np.asarray(value, dtype=float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError(f"{frame}: bad coordinates for {key!r}: {value!r}")
            index = int(match.group(1)) - 1
            if index in out:
                raise ValueError(f"{frame}: {key!r} given twice")
            out[index] = xy
        parsed[frame] = out
    return parsed


def court_error_cm(image_points: np.ndarray, court_points: np.ndarray,
                   probe_image: np.ndarray, probe_court: np.ndarray) -> float | None:
    """Median cm error of ``probe`` mapped through the homography of the points."""
    if len(image_points) < 4:
        return None
    matrix, _ = cv2.findHomography(image_points.astype(np.float32), court_points.astype(np.float32))
    if matrix is None:
        return None
    mapped = cv2.perspectiveTransform(probe_image.reshape(-1, 1, 2).astype(np.float32), matrix)
    return float(np.median(np.linalg.norm(mapped.reshape(-1, 2) - probe_court, axis=1)))


def score_frame(truth: dict[int, np.ndarray], predicted: dict[int, np.ndarray],
                vertices: np.ndarray) -> dict:
    ids = sorted(truth)
    truth_xy = np.array([truth[i] for i in ids])

    points, unlabelled = [], []
    for index, xy in sorted(predicted.items()):
        if index in truth:
            points.append({"kp": index + 1, "xy": xy.tolist(), "reference": truth[index].tolist(),
                           "error_px": float(np.linalg.norm(xy - truth[index]))})
        else:
            # No hand label to compare with: reported, not scored.
            unlabelled.append({"kp": index + 1, "xy": xy.tolist()})

    pred_ids = sorted(predicted)
    return {
        "labelled": len(ids),
        "predicted": len(pred_ids),
        "recall": sum(i in truth for i in pred_ids) / len(ids),
        "points": points,
        "unlabelled": unlabelled,
        # Where the predicted homography puts the hand-labelled points.
        "court_error_cm": court_error_cm(np.array([predicted[i] for i in pred_ids]).reshape(-1, 2),
                                         vertices[pred_ids], truth_xy, vertices[ids]),
        # The same measurement for the hand labels: annotation noise, the floor.
        "human_floor_cm": court_error_cm(truth_xy, vertices[ids], truth_xy, vertices[ids]),
    }


def summarise(frames: dict[str, dict]) -> dict:
    errors = np.array([p["error_px"] for f in frames.values() for p in f["points"]])
    court = [f["court_error_cm"] for f in frames.values() if f["court_error_cm"] is not None]
    floor = [f["human_floor_cm"] for f in frames.values() if f["human_floor_cm"] is not None]
    return {
        "frames": len(frames),
        "frames_without_homography": sum(f["court_error_cm"] is None for f in frames.values()),
        "points": int(errors.size),
        "unlabelled": sum(len(f["unlabelled"]) for f in frames.values()),
        "recall": float(np.mean([f["recall"] for f in frames.values()])) if frames else None,
        "median_px": float(np.median(errors)) if errors.size else None,
        "p90_px": float(np.percentile(errors, 90)) if errors.size else None,
        "close_pct": float(100 * (errors <= CLOSE_PX).mean()) if errors.size else None,
        "gross": int((errors > GROSS_PX).sum()),
        "median_court_cm": float(np.median(court)) if court else None,
        "median_human_floor_cm": float(np.median(floor)) if floor else None,
    }


def draw_overlay(image: np.ndarray, truth: dict[int, np.ndarray], result: dict) -> np.ndarray:
    canvas = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, xy in truth.items():
        centre = tuple(int(round(c)) for c in xy)
        cv2.circle(canvas, centre, 7, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(index + 1), (centre[0] + 8, centre[1] + 16), font, 0.5,
                    (0, 220, 0), 1, cv2.LINE_AA)
    for point in result["points"]:
        xy = tuple(int(round(c)) for c in point["xy"])
        ref = tuple(int(round(c)) for c in point["reference"])
        colour = (0, 0, 255) if point["error_px"] > GROSS_PX else (0, 200, 255)
        cv2.line(canvas, xy, ref, colour, 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, xy, colour, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        cv2.putText(canvas, str(point["kp"]), (xy[0] + 8, xy[1] - 6), font, 0.5, colour, 1,
                    cv2.LINE_AA)
    for point in result["unlabelled"]:
        xy = tuple(int(round(c)) for c in point["xy"])
        cv2.drawMarker(canvas, xy, (255, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        cv2.putText(canvas, str(point["kp"]), (xy[0] + 8, xy[1] - 6), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return add_legend(canvas)


def add_legend(canvas: np.ndarray) -> np.ndarray:
    """A strip under the frame saying which marks are which, so nothing is covered."""
    strip = np.full((44, canvas.shape[1], 3), 30, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 14
    entries = (
        ("circle", (0, 220, 0), "hand label (dataset)"),
        ("cross", (0, 200, 255), f"model, within {GROSS_PX:.0f} px"),
        ("cross", (0, 0, 255), f"model, over {GROSS_PX:.0f} px off (line = to hand label)"),
        ("cross", (255, 255, 255), "model, no hand label to compare"),
    )
    for shape, colour, text in entries:
        if shape == "circle":
            cv2.circle(strip, (x + 7, 22), 7, colour, 2, cv2.LINE_AA)
        else:
            cv2.drawMarker(strip, (x + 7, 22), colour, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        cv2.putText(strip, text, (x + 22, 27), font, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        x += 22 + cv2.getTextSize(text, font, 0.5, 1)[0][0] + 28
    return np.vstack([canvas, strip])


def fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def score(args: argparse.Namespace) -> None:
    run_dir = RUNS_DIR / args.name
    key = json.loads((run_dir / "key.json").read_text())
    predictions_path = args.predictions or run_dir / "task" / "predictions.json"
    predictions = parse_predictions(json.loads(predictions_path.read_text()))
    unknown = set(predictions) - set(key["frames"])
    if unknown:
        raise SystemExit(f"predictions for frames not in the task: {sorted(unknown)}")

    vertices = court_vertices()
    overlay_dir = run_dir / "overlays"
    shutil.rmtree(overlay_dir, ignore_errors=True)
    overlay_dir.mkdir()

    results = {}
    print(f"{'frame':14} {'view':7} {'pred':>4} {'lab':>4} {'recall':>6} {'med px':>7} "
          f"{'gross':>5} {'court cm':>9} {'human cm':>9}")
    for frame, entry in key["frames"].items():
        truth = ground_truth(load_label(Path(entry["label"])))
        result = score_frame(truth, predictions.get(frame, {}), vertices)
        result["middle_court"] = entry["middle_court"]
        result["answered"] = frame in predictions
        results[frame] = result
        errors = [p["error_px"] for p in result["points"]]
        print(f"{frame:14} {'middle' if entry['middle_court'] else 'goal':7} "
              f"{result['predicted']:4d} {result['labelled']:4d} {result['recall']:6.2f} "
              f"{fmt(np.median(errors) if errors else None, '7.1f'):>7} "
              f"{sum(e > GROSS_PX for e in errors):5d} "
              f"{fmt(result['court_error_cm'], '9.0f'):>9} {fmt(result['human_floor_cm'], '9.0f'):>9}")
        image = cv2.resize(cv2.imread(entry["image"]), (FRAME_W, FRAME_H), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(overlay_dir / frame), draw_overlay(image, truth, result))

    groups = {
        "all": results,
        "middle_court": {f: r for f, r in results.items() if r["middle_court"]},
        "goal_in_view": {f: r for f, r in results.items() if not r["middle_court"]},
    }
    summary = {name: summarise(group) for name, group in groups.items()}
    print(f"\n{'':13} {'frames':>6} {'points':>6} {'recall':>6} {'med px':>7} {'p90 px':>7} "
          f"{f'<={CLOSE_PX:.0f}px':>7} {'gross':>5} {'court cm':>9} {'human cm':>9}")
    for name, s in summary.items():
        print(f"{name:13} {s['frames']:6d} {s['points']:6d} {fmt(s['recall'], '6.2f'):>6} "
              f"{fmt(s['median_px'], '7.1f'):>7} {fmt(s['p90_px'], '7.1f'):>7} "
              f"{fmt(s['close_pct'], '6.0f'):>6}% {s['gross']:5d} "
              f"{fmt(s['median_court_cm'], '9.0f'):>9} {fmt(s['median_human_floor_cm'], '9.0f'):>9}")
    unscored = {f: [p["kp"] for p in r["unlabelled"]] for f, r in results.items() if r["unlabelled"]}
    if unscored:
        print("\nnot scored (no hand label): " + "; ".join(
            f"{f} " + ", ".join(f"KP{k:02d}" for k in kps) for f, kps in unscored.items()))
    missing = [f for f, r in results.items() if not r["answered"]]
    if missing:
        print(f"\nno entry for {len(missing)} frame(s): {missing}")
    print(f"\npx at {FRAME_W}x{FRAME_H}; gross = over {GROSS_PX:.0f} px (usually a wrong KP number).\n"
          f"court cm = hand-labelled points mapped through the predicted homography.\n"
          f"overlays: {overlay_dir}  (legend drawn under each frame)")

    (run_dir / "score.json").write_text(json.dumps(
        {"predictions": str(predictions_path), "summary": summary, "frames": results}, indent=2))


# One frame per request: the first run sent all ten in one message and the model
# spent ~190 reasoning tokens a frame.
KEYPOINT_SCHEMA = {
    "type": "object",
    "properties": {"keypoints": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "kp": {"type": "string", "enum": [f"KP{i:02d}" for i in range(1, len(DESCRIPTIONS) + 1)]},
            "x": {"type": "number"},
            "y": {"type": "number"},
        },
        "required": ["kp", "x", "y"],
        "additionalProperties": False,
    }}},
    "required": ["keypoints"],
    "additionalProperties": False,
}

FRAME_PREAMBLE = """\
Two images are attached: image 1 is `template.png`, image 2 is the frame `{frame}`.
Label image 2 only. You cannot run commands or read files in this session; the
instructions below are complete. Where they mention the frames folder or
predictions.json, ignore that: answer with the JSON this session's output schema
asks for -- one entry per landmark you place, {coords}.

---

"""

# How the answer's x/y are expressed. Grounding-trained models often emit
# coordinates normalised to 0..1000 rather than pixels, and Mistral does not
# document which Large 4 uses, so both can be asked for.
COORDS = {
    "pixel": "in pixel coordinates of image 2",
    "norm1000": ("with x and y normalised to 0..1000 of image 2's width and height "
                 "(x=0 left edge, x=1000 right edge, y=0 top, y=1000 bottom) -- this "
                 "replaces the pixel coordinates the instructions below ask for"),
}


def frame_prompt(task_prompt: str, frame: str, coords: str = "pixel") -> str:
    """PROMPT.md for one frame: the task, minus its folder-wide output section."""
    body = task_prompt.split("\n## Output")[0].rstrip()
    return FRAME_PREAMBLE.format(frame=frame, coords=COORDS[coords]) + body + "\n"


def answer_points(answer: dict, coords: str = "pixel") -> dict[str, list[float]]:
    """A schema answer as one frame's ``predictions.json`` entry, in pixels."""
    scale = (FRAME_W / 1000.0, FRAME_H / 1000.0) if coords == "norm1000" else (1.0, 1.0)
    points: dict[str, list[float]] = {}
    for item in answer["keypoints"]:
        if item["kp"] in points:
            raise ValueError(f"{item['kp']} given twice")
        points[item["kp"]] = [float(item["x"]) * scale[0], float(item["y"]) * scale[1]]
    return points


class LabellerStopped(RuntimeError):
    """The labeller gave no answer (limit, outage, timeout); nothing is saved."""


def label_frames(args: argparse.Namespace, labeller: str, ask) -> None:
    """Ask ``ask(task_dir, frame, prompt)`` for each frame, saving as it goes.

    Each answer is written to ``<run>/<labeller>/<frame>.json`` before the next
    request, and a frame that already has one is skipped, so a run cut short by
    a usage limit resumes where it stopped. ``predictions.json`` is rebuilt from
    every saved answer at the end.
    """
    run_dir = RUNS_DIR / args.name
    task_dir = run_dir / "task"
    if not (task_dir / "PROMPT.md").exists():
        raise SystemExit(f"no task at {task_dir}; run prepare first")
    out_dir = run_dir / labeller
    out_dir.mkdir(exist_ok=True)
    task_prompt = (task_dir / "PROMPT.md").read_text()
    coords = getattr(args, "coords", "pixel")
    (out_dir / "prompt_template.txt").write_text(frame_prompt(task_prompt, "{frame}", coords))
    (out_dir / "output_schema.json").write_text(json.dumps(KEYPOINT_SCHEMA, indent=2))

    frames = sorted(path.name for path in (task_dir / "frames").glob("*.jpg"))
    for frame in frames:
        record_path = out_dir / f"{frame}.json"
        if record_path.exists():
            continue
        started = time.monotonic()
        try:
            record = ask(task_dir, frame, frame_prompt(task_prompt, frame, coords))
        except LabellerStopped as error:
            print(f"stopped at {frame}: {error}\nrun the same command again to resume")
            break
        record["coords"] = coords
        record["elapsed_s"] = round(time.monotonic() - started, 1)
        try:
            record["points"] = answer_points(json.loads(record["raw"]), coords)
        except (ValueError, KeyError, TypeError) as error:
            # Answered, but not usably: kept as a result, left out of predictions.
            record["malformed"] = str(error)
        partial = record_path.with_suffix(".tmp")
        partial.write_text(json.dumps(record, indent=2))
        os.replace(partial, record_path)
        usage = record.get("usage") or {}
        reasoning = usage.get("reasoning_output_tokens", usage.get("reasoning_tokens"))
        print(f"{frame}: {len(record.get('points', {})):2d} points, "
              f"{usage.get('output_tokens', usage.get('completion_tokens'))} output tokens"
              + (f" ({reasoning} reasoning)" if reasoning is not None else "")
              + f", {record['elapsed_s']:.0f}s"
              + (f", MALFORMED: {record['malformed']}" if "malformed" in record else "")
              + (f", tried {len(record['commands'])} command(s)" if record.get("commands") else ""),
              flush=True)

    predictions, malformed = {}, []
    for frame in frames:
        record_path = out_dir / f"{frame}.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text())
        if "points" in record:
            predictions[frame] = record["points"]
        else:
            malformed.append(frame)
    (task_dir / "predictions.json").write_text(json.dumps(predictions, indent=1))
    print(f"\n{len(predictions)}/{len(frames)} frames answered"
          + (f"; malformed: {malformed}" if malformed else "")
          + f"\n  python -m scripts.vlm_court_keypoints score --name {args.name}")


def find_codex() -> str:
    found = shutil.which("codex")
    if found:
        return found
    # The VS Code extension ships its own binary and does not put it on PATH.
    candidates = sorted(Path.home().glob(".vscode-server/extensions/openai.chatgpt-*/bin/*/codex"),
                        key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit("codex CLI not found; pass --codex")
    return str(candidates[-1])


def codex(args: argparse.Namespace) -> None:
    binary = args.codex or find_codex()
    schema_path = RUNS_DIR / args.name / "codex" / "output_schema.json"

    def ask(task_dir: Path, frame: str, prompt: str) -> dict:
        last = schema_path.parent / f"{frame}.last.txt"
        images = f"{task_dir / 'template.png'},{task_dir / 'frames' / frame}"
        # read-only: the labeller has nothing to run. (Codex's bubblewrap sandbox
        # also fails inside some outer sandboxes, so every command would fail anyway.)
        command = [binary, "exec", "-m", args.model, "-c", f'model_reasoning_effort="{args.effort}"',
                   "-s", "read-only", "-C", str(task_dir), "--json", "--output-schema",
                   str(schema_path), "-o", str(last), "-i", images, "-"]
        try:
            done = subprocess.run(command, input=prompt, capture_output=True, text=True,
                                  timeout=args.timeout)
        except subprocess.TimeoutExpired as error:
            raise LabellerStopped(f"no answer in {args.timeout:.0f}s") from error
        events = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
        usage = next((e["usage"] for e in reversed(events) if e.get("type") == "turn.completed"), None)
        text = last.read_text() if last.exists() else ""
        last.unlink(missing_ok=True)
        if done.returncode != 0 or usage is None or not text.strip():
            # Usually the plan's usage limit.
            errors = [e for e in events if e.get("type") in ("error", "turn.failed")]
            raise LabellerStopped(f"codex exited {done.returncode}: "
                                  f"{json.dumps(errors)[:600]} {done.stderr[-600:]}")
        commands = [e["item"].get("command") for e in events
                    if e.get("type") == "item.completed"
                    and (e.get("item") or {}).get("type") == "command_execution"]
        return {"model": args.model, "effort": args.effort, "usage": usage,
                "commands": commands, "raw": text}

    label_frames(args, "codex", ask)


MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


def mistral_payload(model: str, prompt: str, images: list[Path], extra: dict) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": image_data_url(path)} for path in images
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "court_keypoints", "schema": KEYPOINT_SCHEMA, "strict": True}},
    }
    payload.update(extra)
    return payload


def mistral(args: argparse.Namespace) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit("MISTRAL_API_KEY is not set (environment or .env)")
    extra = json.loads(args.extra_body) if args.extra_body else {}

    def ask(task_dir: Path, frame: str, prompt: str) -> dict:
        payload = mistral_payload(args.model, prompt,
                                  [task_dir / "template.png", task_dir / "frames" / frame], extra)
        request = Request(MISTRAL_URL, data=json.dumps(payload).encode(), method="POST",
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {key}"})
        for attempt in range(args.retries + 1):
            try:
                with urlopen(request, timeout=args.timeout) as response:
                    body = json.load(response)
                break
            except HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:600]
                # 429 and 5xx are worth a wait; anything else will not fix itself.
                if (error.code == 429 or error.code >= 500) and attempt < args.retries:
                    time.sleep(min(2 ** (attempt + 2), 60))
                    continue
                raise LabellerStopped(f"HTTP {error.code}: {detail}") from error
            except (URLError, TimeoutError) as error:
                if attempt < args.retries:
                    time.sleep(min(2 ** (attempt + 2), 60))
                    continue
                raise LabellerStopped(f"could not reach the API: {error}") from error
        return {"model": body.get("model", args.model), "extra_body": extra,
                "usage": body.get("usage", {}), "raw": extract_content(body)}

    label_frames(args, "mistral", ask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="write the task folder for the labeller")
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--split", default="test")
    p.add_argument("--middle", type=int, default=5, help="frames with no goal line labelled")
    p.add_argument("--other", type=int, default=5, help="frames with a goal line labelled")
    p.set_defaults(func=prepare)

    s = sub.add_parser("score", help="grade predictions.json against the hand labels")
    s.add_argument("--name", required=True)
    s.add_argument("--predictions", type=Path, default=None)
    s.set_defaults(func=score)

    c = sub.add_parser("codex", help="label the task with GPT-6 Astra via the Codex CLI, per frame")
    c.add_argument("--name", required=True)
    c.add_argument("--model", default="gpt-6-astra")
    c.add_argument("--effort", default="high", help="Codex model_reasoning_effort")
    c.add_argument("--codex", default=None, help="path to the codex binary")
    c.add_argument("--timeout", type=float, default=1200.0, help="seconds per frame")
    c.set_defaults(func=codex)

    m = sub.add_parser("mistral", help="label the task through the Mistral API, per frame")
    m.add_argument("--name", required=True)
    m.add_argument("--model", default="mistral-large-4",
                   help="API model id; GET https://api.mistral.ai/v1/models lists them")
    m.add_argument("--coords", choices=sorted(COORDS), default="pixel",
                   help="ask for pixel or 0..1000-normalised coordinates")
    m.add_argument("--extra-body", default=None,
                   help='JSON merged into the request, e.g. \'{"temperature": 0}\'')
    m.add_argument("--timeout", type=float, default=600.0, help="seconds per request")
    m.add_argument("--retries", type=int, default=3, help="retries on 429/5xx/network")
    m.set_defaults(func=mistral)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
