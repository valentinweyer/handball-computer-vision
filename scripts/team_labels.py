"""Ground-truth team labels, so team classification can finally be scored on
accuracy instead of self-consistency.

`compare_trackers.team_purity_%` compares each per-frame prediction against a
run's own majority vote -- it never touches ground truth, so a classifier that
stably split players by height would score ~100% too. This module produces
real labels to fix that.

Substrate: one reference run's `(frame_idx, column)` pairs. Boxes come from
`run["boxes"]`, masks come from the SAM2 mask cache that same run wrote, and
labels are keyed the same way -- so fitting, labelling and scoring all see
identical pixels (the thing v1 of team_model.py got wrong by fitting on one
crop geometry and predicting on another).

There is no display in this environment (checked: DISPLAY is empty, this is a
TTY session) so cv2.imshow / matplotlib GUIs are out. Two labelling paths exist:

  1. `sheets` (primary -- works with no browser at all): numbered PNG contact
     sheets, shown inline in the chat the same way every other image this
     session has been. Reply with one character per crop, in reading order
     (left-to-right, top-to-bottom, sheet 1 then sheet 2 ...):
     H=HAN, B=BER, K=GK, X=skip/unclear. Then `savecodes`.
  2. `sample` (needs a browser/webview): self-contained HTML page, crops
     inlined as base64 JPEG, click-to-cycle labels, a live JSON textarea.
     Then `save` with the copied JSON.

Usage:
    python team_labels.py sheets                       # writes numbered PNG sheets
    python team_labels.py savecodes 'HBBKX...'          # all sheets concatenated, one string
    python team_labels.py savecodes --file codes.txt    # same, from a file

    python team_labels.py sample                        # writes the HTML labelling page
    python team_labels.py save '<pasted JSON>'           # writes the labels file
    python team_labels.py save --file labels.json        # same, from a file
"""
import argparse
import os
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from handball_cv.tracking.mask_cache import MaskCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_VIDEO_PATH = Path(os.getenv("HANDBALL_CV_VIDEO", PROJECT_ROOT / "data/raw/Han-Ber4.mp4"))
RUN_DIR = Path(os.getenv("HANDBALL_CV_RUN_DIR", PROJECT_ROOT / "runs"))
MASK_CACHE_DIR = PROJECT_ROOT / "data/cache/masks" / SOURCE_VIDEO_PATH.stem / "sam2"
LABELS_PATH = PROJECT_ROOT / "data/annotations/team" / f"{SOURCE_VIDEO_PATH.stem}-frame-labels.json"
LABEL_PAGE_PATH = PROJECT_ROOT / "runs/labeling" / SOURCE_VIDEO_PATH.stem / "label_teams.html"

REFERENCE_RUN = "sam2_identity_masked_smoke"
N_SAMPLES = 200
SEED = 0
# Crop shown to the human labeller -- deliberately more generous than the
# model's own training crop (CENTERED_CROP_SCALE_W/H = 0.4/0.4 in
# team_model.py): a person needs more context than SigLIP does to tell HAN
# from BER from a goalkeeper confidently.
DISPLAY_SCALE = 0.7
LABEL_OPTIONS = ["HAN", "BER", "GK", "SKIP"]

# ── sheets (no-browser path) ─────────────────────────────────────────────────
SHEET_DIR = PROJECT_ROOT / "runs/labeling" / SOURCE_VIDEO_PATH.stem
SHEET_ORDER_PATH = SHEET_DIR / "sheet_order.json"
GRID_COLS = 5
CELLS_PER_SHEET = 25  # 5x5
CELL_PX = 140         # crop area per cell, square
LABEL_BAR_PX = 26     # header strip per cell, holds the index number
CODE_MAP = {"H": "HAN", "B": "BER", "K": "GK", "X": None}


def load_run(name: str) -> dict:
    with np.load(RUN_DIR / f"{name}.npz", allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def sample_pairs(run: dict, n_samples: int, seed: int) -> list:
    """Stratified (frame_idx, col) sample: spread across tracks and across the
    clip, not wherever the model happened to be most confident -- ground truth
    must not inherit the model's own blind spots."""
    boxes = run["boxes"]  # (T, P, 4)
    T, P = boxes.shape[0], boxes.shape[1]
    valid = np.isfinite(boxes).all(axis=2)  # (T, P)

    rng = np.random.default_rng(seed)
    per_col = max(n_samples // max(P, 1), 1)

    pairs = []
    for col in range(P):
        frames = np.nonzero(valid[:, col])[0]
        if len(frames) == 0:
            continue
        chosen = rng.choice(frames, size=min(per_col, len(frames)), replace=False)
        pairs.extend((int(f), col) for f in chosen)

    rng.shuffle(pairs)
    return pairs[:n_samples]


def to_data_uri(crop_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", crop_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def collect_crops(run: dict, pairs: list) -> list:
    """[(key, crop_rgb), ...] for every (frame, col) pair -- shared substrate
    for both labelling paths so `sheets` and `sample` see identical pixels."""
    boxes = run["boxes"]
    source = str(run["source"])
    by_frame: dict = {}
    for fid, col in pairs:
        by_frame.setdefault(fid, []).append(col)

    items = []  # (key, crop_rgb)
    for t, frame_bgr in enumerate(sv.get_video_frames_generator(source)):
        idx = t - 1  # boxes row idx == source frame idx+1, established convention
        if idx not in by_frame:
            continue
        frame_rgb = frame_bgr[:, :, ::-1]
        cols = by_frame[idx]
        row = boxes[idx]
        scaled = sv.scale_boxes(xyxy=row[cols], factor=DISPLAY_SCALE)
        for i, col in enumerate(cols):
            crop = sv.crop_image(frame_rgb, scaled[i])
            if crop.size == 0:
                continue
            items.append((f"{idx}_{col}", crop))
    return items


def build_label_sheets(run: dict, pairs: list, out_dir: Path) -> int:
    """Numbered PNG contact sheets -- the no-browser labelling path.

    Presentation order is randomised (not sorted by prediction confidence,
    which would anchor the labeller toward the model's own blind spots) and
    that exact order is persisted to SHEET_ORDER_PATH, so `parse_codes` can
    map the human's typed response (position N -> key) without re-deriving
    anything that could drift out of sync with what was actually rendered.
    """
    items = collect_crops(run, pairs)
    rng = np.random.default_rng(SEED + 1)
    order = rng.permutation(len(items))
    items = [items[i] for i in order]

    out_dir.mkdir(parents=True, exist_ok=True)
    keys_in_order = [k for k, _ in items]
    SHEET_ORDER_PATH.write_text(json.dumps(keys_in_order))

    n_sheets = (len(items) + CELLS_PER_SHEET - 1) // CELLS_PER_SHEET
    cell_h = CELL_PX + LABEL_BAR_PX
    for s in range(n_sheets):
        chunk = items[s * CELLS_PER_SHEET:(s + 1) * CELLS_PER_SHEET]
        rows = (len(chunk) + GRID_COLS - 1) // GRID_COLS
        sheet = np.full((rows * cell_h, GRID_COLS * CELL_PX, 3), 30, dtype=np.uint8)
        for i, (_key, crop) in enumerate(chunk):
            r, c = divmod(i, GRID_COLS)
            inner = CELL_PX - 6
            resized = cv2.resize(crop, (inner, inner))
            y0 = r * cell_h + LABEL_BAR_PX
            x0 = c * CELL_PX + 3
            sheet[y0:y0 + inner, x0:x0 + inner] = resized[:, :, ::-1]
            global_idx = s * CELLS_PER_SHEET + i + 1
            cv2.putText(sheet, str(global_idx), (c * CELL_PX + 6, r * cell_h + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 220, 60), 2, cv2.LINE_AA)
        path = out_dir / f"label_sheet_{s + 1:02d}.png"
        cv2.imwrite(str(path), sheet)
        lo, hi = s * CELLS_PER_SHEET + 1, s * CELLS_PER_SHEET + len(chunk)
        print(f"wrote {path}  (crops {lo}-{hi} of {len(items)})")
    return n_sheets


def parse_codes(codes: str) -> dict:
    """Compact reply ('HBBKX...', one char per crop, reading order across all
    sheets) -> {key: label}. Whitespace/newlines/commas between codes are
    tolerated since a human is very likely to introduce them when typing."""
    keys = json.loads(SHEET_ORDER_PATH.read_text())
    cleaned = "".join(ch for ch in codes.upper() if ch not in " ,\n\t")
    if len(cleaned) != len(keys):
        raise SystemExit(
            f"expected {len(keys)} codes (one per crop across all sheets), got "
            f"{len(cleaned)} -- run `sheets` again if the count seems wrong"
        )
    labels = {}
    for key, ch in zip(keys, cleaned):
        if ch not in CODE_MAP:
            raise SystemExit(f"unknown code {ch!r} (key {key}) -- valid: {sorted(CODE_MAP)}")
        mapped = CODE_MAP[ch]
        if mapped is not None:
            labels[key] = mapped
    return labels


def build_label_page(run: dict, pairs: list, out_path: Path) -> None:
    boxes = run["boxes"]
    source = str(run["source"])
    by_frame: dict = {}
    for fid, col in pairs:
        by_frame.setdefault(fid, []).append(col)

    mask_cache = MaskCache.read_only(MASK_CACHE_DIR)
    items = []  # (key, data_uri, has_mask)

    for t, frame_bgr in enumerate(sv.get_video_frames_generator(source)):
        idx = t - 1  # boxes row idx == source frame idx+1, established convention
        if idx not in by_frame:
            continue
        frame_rgb = frame_bgr[:, :, ::-1]
        cols = by_frame[idx]
        row = boxes[idx]
        scaled = sv.scale_boxes(xyxy=row[cols], factor=DISPLAY_SCALE)
        masks = mask_cache.load(idx, np.array(cols), frame_rgb.shape[:2])
        for i, col in enumerate(cols):
            crop = sv.crop_image(frame_rgb, scaled[i])
            if crop.size == 0:
                continue
            has_mask = masks is not None and masks[i].any()
            items.append((f"{idx}_{col}", to_data_uri(crop), has_mask))

    rng = np.random.default_rng(SEED + 1)
    order = rng.permutation(len(items))  # randomised presentation order
    items = [items[i] for i in order]

    n_with_mask = sum(1 for _, _, m in items if m)
    print(f"{len(items)} crops collected ({n_with_mask} with a cached mask)")

    def buttons_html(key: str) -> str:
        parts = []
        for lbl in LABEL_OPTIONS:
            onclick = "setLabel(" + repr(key) + ", " + repr(lbl) + ", this)"
            parts.append(f'<button data-label="{lbl}" onclick="{onclick}">{lbl}</button>')
        return "".join(parts)

    cards = []
    for key, data_uri, has_mask in items:
        badge = "" if has_mask else '<span class="nomask">no cached mask</span>'
        cards.append(f"""
        <div class="card" data-key="{key}">
          <img src="{data_uri}">
          {badge}
          <div class="buttons">
            {buttons_html(key)}
          </div>
        </div>""")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Team labelling -- {REFERENCE_RUN}</title>
<style>
  body {{ font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }}
  h1 {{ font-size: 16px; }}
  #progress {{ position: sticky; top: 0; background: #111; padding: 8px 0; z-index: 10; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .card {{ background: #222; border: 2px solid #444; border-radius: 6px; padding: 6px; width: 130px; text-align: center; }}
  .card.done {{ border-color: #4caf50; }}
  .card img {{ width: 118px; height: auto; image-rendering: auto; border-radius: 4px; }}
  .nomask {{ display: block; color: #f0ad4e; font-size: 10px; }}
  .buttons {{ display: flex; flex-wrap: wrap; gap: 3px; justify-content: center; margin-top: 4px; }}
  button {{ font-size: 11px; padding: 3px 6px; cursor: pointer; border-radius: 3px; border: 1px solid #555; background: #333; color: #eee; }}
  button.selected {{ background: #4caf50; color: #111; font-weight: bold; }}
  textarea {{ width: 100%; height: 120px; margin-top: 12px; background: #000; color: #0f0; font-family: monospace; }}
</style></head><body>
<div id="progress"><h1>{REFERENCE_RUN}: {len(items)} crops. Click a label on each card, then copy the JSON below.</h1>
<div id="count">0 / {len(items)} labelled</div></div>
<div class="grid">
{''.join(cards)}
</div>
<h2>JSON (copy this back)</h2>
<textarea id="out" readonly></textarea>
<script>
const labels = {{}};
function render() {{
  document.getElementById('out').value = JSON.stringify(labels);
  document.getElementById('count').innerText = Object.keys(labels).length + ' / {len(items)} labelled';
}}
function setLabel(key, label, btn) {{
  labels[key] = label;
  const card = btn.closest('.card');
  card.classList.add('done');
  card.querySelectorAll('button').forEach(b => b.classList.toggle('selected', b === btn));
  render();
}}
</script>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sheets")
    codes_p = sub.add_parser("savecodes")
    codes_p.add_argument("codes", nargs="?", default=None)
    codes_p.add_argument("--file", type=Path, default=None)
    sub.add_parser("sample")
    save_p = sub.add_parser("save")
    save_p.add_argument("json_text", nargs="?", default=None)
    save_p.add_argument("--file", type=Path, default=None)
    args = ap.parse_args()

    if args.cmd == "sheets":
        run = load_run(REFERENCE_RUN)
        pairs = sample_pairs(run, N_SAMPLES, SEED)
        n_sheets = build_label_sheets(run, pairs, SHEET_DIR)
        print(f"wrote {n_sheets} sheet(s) to {SHEET_DIR}")
        print("label order saved to", SHEET_ORDER_PATH)
        print("reply with one code per crop, reading order, sheet 1 then sheet 2 ...:")
        print("  H=HAN  B=BER  K=GK  X=skip/unclear")
        print("then run: python team_labels.py savecodes '<your codes>'")

    elif args.cmd == "savecodes":
        if args.file is not None:
            codes = args.file.read_text()
        elif args.codes is not None:
            codes = args.codes
        else:
            raise SystemExit("pass a code string or --file")
        labels = parse_codes(codes)
        if not labels:
            raise SystemExit("no labels parsed (all X/skip?) -- nothing to save")
        LABELS_PATH.write_text(json.dumps(
            {"reference_run": REFERENCE_RUN, "labels": labels}, indent=2))
        counts = {}
        for v in labels.values():
            counts[v] = counts.get(v, 0) + 1
        print(f"saved {len(labels)} labels -> {LABELS_PATH}")
        print("counts:", counts)

    elif args.cmd == "sample":
        run = load_run(REFERENCE_RUN)
        pairs = sample_pairs(run, N_SAMPLES, SEED)
        build_label_page(run, pairs, LABEL_PAGE_PATH)
        print(f"wrote {LABEL_PAGE_PATH}")
        print("open it, label the crops, then run:")
        print("  python team_labels.py save '<pasted JSON>'")

    elif args.cmd == "save":
        if args.file is not None:
            text = args.file.read_text()
        elif args.json_text is not None:
            text = args.json_text
        else:
            raise SystemExit("pass JSON text or --file")
        data = json.loads(text)
        if not data:
            raise SystemExit("no labels in input -- did you click any buttons before copying?")
        LABELS_PATH.write_text(json.dumps(
            {"reference_run": REFERENCE_RUN, "labels": data}, indent=2))
        counts = {}
        for v in data.values():
            counts[v] = counts.get(v, 0) + 1
        print(f"saved {len(data)} labels -> {LABELS_PATH}")
        print("counts:", counts)


if __name__ == "__main__":
    main()
