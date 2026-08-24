"""Compact tracklet-level team labelling.

Instead of asking for hundreds of crop labels, this renders one montage row per
track with several temporally separated, automatically selected views. The
human replies with one character per row:

    A = team A       B = team B       K = goalkeeper
    O = other        M = mixed ID     X = unclear

The displayed samples are expanded into the frame/column label format consumed
by compare_trackers.py. Mixed, other and unclear rows are retained in the
track-level file but excluded from per-frame team accuracy.

Usage:
    python label_tracklets.py sheets --run sam2_identity_ref --team-a HAN --team-b BER
    python label_tracklets.py savecodes ABBAK...
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "source"
RUN_DIR = SOURCE_DIR / ".runs"
OUT_DIR = SOURCE_DIR / "team_classification_examples"

CELL_PX = 130
ROW_PX = 156
LABEL_W = 150
HEADER_H = 62
VALID_CODES = {"A", "B", "K", "O", "M", "X"}


def load_run(name: str) -> dict:
    with np.load(RUN_DIR / f"{name}.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def output_paths(source: str) -> tuple[Path, Path, Path]:
    video = Path(source)
    order = video.parent / f".{video.stem}_track_label_sheet_order.json"
    tracks = video.parent / f".{video.stem}_track_team_labels.json"
    frames = video.parent / f".{video.stem}_team_labels.json"
    return order, tracks, frames


def fit_in_cell(image: np.ndarray, size: int) -> np.ndarray:
    """Letterbox a BGR crop into a square without distorting the jersey."""
    output = np.full((size, size, 3), 24, dtype=np.uint8)
    if image.size == 0:
        return output
    height, width = image.shape[:2]
    scale = min(size / max(width, 1), size / max(height, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    x0 = (size - new_width) // 2
    y0 = (size - new_height) // 2
    output[y0:y0 + new_height, x0:x0 + new_width] = resized
    return output


def select_examples(run: dict, samples_per_track: int) -> list[dict]:
    """Pick clear, temporally spread crops for every track column.

    Valid rows are split into temporal bins. Within each bin, the
    largest/sharpest crop wins, with an overlap penalty so solo views are
    preferred over scrums. Run row r corresponds to source frame r + 1.
    """
    boxes = run["boxes"]
    track_ids = run.get("track_ids", np.arange(boxes.shape[1]) + 1)
    is_goalkeeper = run.get(
        "is_goalkeeper", np.zeros(boxes.shape[1], dtype=bool))
    track_count = boxes.shape[1]

    row_to_bin = []
    for column in range(track_count):
        valid_rows = np.nonzero(
            np.isfinite(boxes[:, column]).all(axis=1))[0]
        mapping = {}
        for bin_index, chunk in enumerate(
                np.array_split(valid_rows, samples_per_track)):
            for row_index in chunk:
                mapping[int(row_index)] = bin_index
        row_to_bin.append(mapping)

    best = [[None] * samples_per_track for _ in range(track_count)]
    for source_frame, frame_bgr in enumerate(
            sv.get_video_frames_generator(str(run["source"]))):
        row_index = source_frame - 1
        if row_index < 0 or row_index >= len(boxes):
            continue
        row = boxes[row_index]
        valid_columns = np.nonzero(np.isfinite(row).all(axis=1))[0]
        for column in valid_columns:
            bin_index = row_to_bin[int(column)].get(row_index)
            if bin_index is None:
                continue

            display_box = sv.scale_boxes(
                row[column][None], factor=0.82)[0]
            crop = sv.crop_image(frame_bgr, display_box)
            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(
                cv2.Laplacian(gray, cv2.CV_64F).var())
            width = row[column, 2] - row[column, 0]
            height = row[column, 3] - row[column, 1]
            area = float(width * height)

            other_columns = valid_columns[valid_columns != column]
            max_overlap = 0.0
            if len(other_columns):
                max_overlap = float(
                    sv.box_iou_batch(
                        row[column][None], row[other_columns]).max())

            score = np.log1p(max(area, 0.0))
            score *= 1.0 + min(sharpness / 300.0, 1.0)
            score *= max(0.2, 1.0 - max_overlap)
            current = best[int(column)][bin_index]
            if current is None or score > current[0]:
                best[int(column)][bin_index] = (
                    score, row_index, crop.copy())

    examples = []
    for column in range(track_count):
        selected = [item for item in best[column] if item is not None]
        examples.append({
            "column": column,
            "track_id": int(track_ids[column]),
            "detector_goalkeeper": bool(is_goalkeeper[column]),
            "samples": [
                (int(row_index), crop)
                for _score, row_index, crop in selected
            ],
        })
    return examples


def build_sheets(
    run: dict,
    run_name: str,
    team_a: str,
    team_b: str,
    samples_per_track: int,
    rows_per_sheet: int,
) -> list[Path]:
    examples = select_examples(run, samples_per_track)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet_width = LABEL_W + samples_per_track * CELL_PX
    sheet_count = (
        len(examples) + rows_per_sheet - 1) // rows_per_sheet
    written = []

    for sheet_index in range(sheet_count):
        start = sheet_index * rows_per_sheet
        chunk = examples[start:start + rows_per_sheet]
        sheet = np.full(
            (HEADER_H + len(chunk) * ROW_PX, sheet_width, 3),
            24,
            dtype=np.uint8,
        )
        legend = (
            f"A={team_a}  B={team_b}  K=keeper  O=other  "
            "M=mixed ID  X=unclear"
        )
        cv2.putText(
            sheet, legend, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(
            sheet, "Reply with one code per numbered row", (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 220, 80), 1,
            cv2.LINE_AA)

        for local_row, item in enumerate(chunk):
            global_row = start + local_row + 1
            y0 = HEADER_H + local_row * ROW_PX
            if local_row:
                cv2.line(
                    sheet, (0, y0), (sheet_width, y0),
                    (70, 70, 70), 1)
            suffix = (
                "  detector:GK" if item["detector_goalkeeper"] else "")
            cv2.putText(
                sheet,
                f"{global_row:02d}  track {item['track_id']}{suffix}",
                (8, y0 + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (80, 220, 80),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                sheet, f"column {item['column']}", (8, y0 + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1,
                cv2.LINE_AA)

            for sample_index, (row_index, crop) in enumerate(
                    item["samples"]):
                cell = fit_in_cell(crop, CELL_PX - 10)
                x0 = LABEL_W + sample_index * CELL_PX + 5
                sheet[
                    y0 + 5:y0 + CELL_PX - 5,
                    x0:x0 + CELL_PX - 10,
                ] = cell
                cv2.putText(
                    sheet, f"f{row_index + 1}", (x0 + 3, y0 + CELL_PX + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180),
                    1, cv2.LINE_AA)

        path = OUT_DIR / f"track_label_sheet_{sheet_index + 1:02d}.png"
        if not cv2.imwrite(str(path), sheet):
            raise RuntimeError(f"failed to write {path}")
        written.append(path)

    order_path, _track_path, _frame_path = output_paths(str(run["source"]))
    order_path.write_text(json.dumps({
        "schema_version": 1,
        "reference_run": run_name,
        "source": str(run["source"]),
        "team_a": team_a,
        "team_b": team_b,
        "items": [{
            "column": item["column"],
            "track_id": item["track_id"],
            "detector_goalkeeper": item["detector_goalkeeper"],
            "sample_rows": [
                row_index for row_index, _crop in item["samples"]],
        } for item in examples],
    }, indent=2))
    return written


def save_codes(source: str, codes: str) -> tuple[Path, Path, dict]:
    order_path, track_path, frame_path = output_paths(source)
    order = json.loads(order_path.read_text())
    cleaned = "".join(
        character for character in codes.upper()
        if character not in " ,\n\t"
    )
    items = order["items"]
    if len(cleaned) != len(items):
        raise SystemExit(
            f"expected {len(items)} codes, got {len(cleaned)}")
    invalid = sorted(set(cleaned) - VALID_CODES)
    if invalid:
        raise SystemExit(
            f"unknown codes {invalid}; valid: {sorted(VALID_CODES)}")

    track_labels = {}
    frame_labels = {}
    counts = {}
    for code, item in zip(cleaned, items):
        label = {
            "A": order["team_a"],
            "B": order["team_b"],
            "K": "GK",
            "O": "OTHER",
            "M": "MIXED",
            "X": "UNCLEAR",
        }[code]
        counts[label] = counts.get(label, 0) + 1
        track_labels[str(item["track_id"])] = {
            "column": item["column"],
            "code": code,
            "label": label,
        }
        if label in {order["team_a"], order["team_b"], "GK"}:
            for row_index in item["sample_rows"]:
                frame_labels[
                    f"{row_index}_{item['column']}"] = label

    track_path.write_text(json.dumps({
        "reference_run": order["reference_run"],
        "team_a": order["team_a"],
        "team_b": order["team_b"],
        "tracks": track_labels,
    }, indent=2))
    frame_path.write_text(json.dumps({
        "reference_run": order["reference_run"],
        "labels": frame_labels,
    }, indent=2))
    return track_path, frame_path, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    sheets = commands.add_parser("sheets")
    sheets.add_argument("--run", default="sam2_identity_ref")
    sheets.add_argument("--team-a", default="HAN")
    sheets.add_argument("--team-b", default="BER")
    sheets.add_argument("--samples", type=int, default=6)
    sheets.add_argument("--rows-per-sheet", type=int, default=7)

    save = commands.add_parser("savecodes")
    save.add_argument("codes", nargs="?")
    save.add_argument("--file", type=Path)
    save.add_argument(
        "--source", default=str(SOURCE_DIR / "Han-Ber4.mp4"))
    args = parser.parse_args()

    if args.command == "sheets":
        run = load_run(args.run)
        paths = build_sheets(
            run,
            args.run,
            args.team_a,
            args.team_b,
            args.samples,
            args.rows_per_sheet,
        )
        print(f"wrote {len(paths)} compact sheets")
        for path in paths:
            print(path)
        print("reply with one code per row across all sheets")
    else:
        if args.file:
            codes = args.file.read_text()
        elif args.codes:
            codes = args.codes
        else:
            raise SystemExit("pass codes or --file")
        track_path, frame_path, counts = save_codes(args.source, codes)
        print(f"saved track labels -> {track_path}")
        print(f"saved displayed-frame labels -> {frame_path}")
        print("counts:", counts)


if __name__ == "__main__":
    main()
