"""Contact sheets for reviewing mined identities and digit masking by eye.

Identity sheets: one per (video, team, number), one row per accepted fragment,
crops spread across the fragment's life, unmasked, with number-detector boxes
outlined in red. The legend says the label is mined from that fragment's own
OCR reads, so a reviewer checks the shirt and the person, not the model.

Masking sheets: a seeded random sample of crops after `digits` and `strict`
masking, numbered, for counting digits that remain visible.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from experiments.teammate_reid.masking import apply_mask

CELL_H = 150
PER_ROW = 8
LABEL_W = 300


def _crop(out: Path, r: dict) -> tuple[np.ndarray, list]:
    img = cv2.imread(str(out / "crops" / f"{r['crop_id']}.jpg"))
    x1, y1 = np.asarray(r["box"], dtype=float).round().astype(int)[:2]
    x1, y1 = max(0, x1), max(0, y1)
    local = [(a - x1, b - y1, c - x1, d - y1) for a, b, c, d in r["digit_boxes"]]
    return img, local


def _fit(img: np.ndarray, h: int = CELL_H) -> np.ndarray:
    scale = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), h))


def _text(img, text, org, scale=0.5, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _row(cells: list, label_lines: list, width: int) -> np.ndarray:
    row = np.full((CELL_H + 8, width, 3), 40, np.uint8)
    for i, line in enumerate(label_lines):
        _text(row, line, (6, 20 + 18 * i), 0.45)
    x = LABEL_W
    for cell in cells:
        if x + cell.shape[1] > width:
            break
        row[4:4 + CELL_H, x:x + cell.shape[1]] = cell
        x += cell.shape[1] + 4
    return row


def _legend(width: int, lines: list) -> np.ndarray:
    head = np.full((22 * len(lines) + 10, width, 3), 20, np.uint8)
    for i, line in enumerate(lines):
        _text(head, line, (6, 20 + 22 * i), 0.55, (180, 255, 255))
    return head


def render_sheets(out: Path) -> None:
    out = Path(out)
    crops = json.loads((out / "crops.json").read_text())
    fragments = {f["key"]: f for f in json.loads((out / "fragments.json").read_text())}
    by_identity: dict = {}
    for r in crops:
        by_identity.setdefault(r["identity"], {}).setdefault(r["fragment"], []).append(r)
    width = LABEL_W + PER_ROW * (CELL_H // 2 + 30)
    sheet_dir = out / "sheets"
    sheet_dir.mkdir(exist_ok=True)
    index = []
    for identity, frags in sorted(by_identity.items()):
        rows = [_legend(width, [
            f"{identity}   label = this fragment's OWN OCR reads (mined, not model output)",
            "red box = number-detector box (hidden in training/eval).  Row = one source-track fragment.",
        ])]
        for n, (key, rs) in enumerate(sorted(frags.items(), key=lambda kv: fragments[kv[0]]["start"])):
            f = fragments[key]
            pick = [rs[i] for i in np.linspace(0, len(rs) - 1, min(PER_ROW, len(rs))).round().astype(int)]
            cells = []
            for r in pick:
                img, local = _crop(out, r)
                for a, b, c, d in local:
                    cv2.rectangle(img, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 1)
                cell = _fit(img)
                _text(cell, str(r["frame"]), (2, CELL_H - 6), 0.4)
                cells.append(cell)
            reads = ",".join(f"{v}x{c}" for v, c in sorted(f["reads"].items(), key=lambda kv: -kv[1])[:4])
            rows.append(_row(cells, [f"F{n} track {f['track_id']}", f"frames {f['start']}-{f['end']}",
                                     f"opened by {f['opened_by']}", f"reads {reads}",
                                     f"crops {len(rs)}"], width))
            index.append({"identity": identity, "row": f"F{n}", "fragment": key})
        name = identity.replace("|", "_").replace("#", "n")
        path = sheet_dir / f"{name}.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (sheet_dir / "index.json").write_text(json.dumps(index, indent=1))

    rng = np.random.default_rng(0)
    sample = rng.choice(len(crops), size=min(96, len(crops)), replace=False)
    (sheet_dir / "masking_audit_sample.json").write_text(json.dumps(
        [int(crops[int(i)]["crop_id"]) for i in sample]))
    per_page, sheet_w, cell_h = 24, 1800, 300
    for mode in ("digits", "strict"):
        for page in range(0, len(sample), per_page):
            cells = []
            for k in range(page, min(page + per_page, len(sample))):
                r = crops[int(sample[k])]
                img, local = _crop(out, r)
                masked, _ = apply_mask(img[:, :, ::-1], local, mode, seed_key=str(r["crop_id"]))
                cell = _fit(np.ascontiguousarray(masked[:, :, ::-1]), cell_h)
                if cell.shape[1] > sheet_w // 4:
                    cell = cv2.resize(cell, (sheet_w // 4, int(cell_h * (sheet_w // 4) / cell.shape[1])))
                _text(cell, f"{k}", (2, 18), 0.6, (0, 255, 255))
                cells.append(cell)
            rows, line, x = [], [], 0
            for cell in cells:
                if x + cell.shape[1] > sheet_w:
                    rows.append(line)
                    line, x = [], 0
                line.append(cell)
                x += cell.shape[1] + 4
            rows.append(line)
            canvas = [_legend(sheet_w, [f"Digit masking audit, mode={mode}: grey = hidden region. "
                                        "Count crops where any printed digit is still legible.",
                                        "Yellow numbers = index into masking_audit_sample.json (seed 0)."])]
            for line in rows:
                strip = np.full((cell_h + 4, sheet_w, 3), 40, np.uint8)
                x = 0
                for cell in line:
                    strip[2:2 + cell.shape[0], x:x + cell.shape[1]] = cell
                    x += cell.shape[1] + 4
                canvas.append(strip)
            cv2.imwrite(str(sheet_dir / f"masking_audit_{mode}_{page // per_page}.jpg"),
                        np.vstack(canvas), [cv2.IMWRITE_JPEG_QUALITY, 90])


def render_examples(out: Path, fold: str, tag: str, n: int = 5) -> Path:
    """Query vs proposed identity for the most confident correct and wrong
    proposals of one configuration (unmasked crops, for a human to judge)."""
    out = Path(out)
    rows = json.loads((out / f"rows_{fold}.json").read_text())[tag]
    crops = {c["crop_id"]: c for c in json.loads((out / "crops.json").read_text())}
    by_fragment, by_identity = {}, {}
    for c in sorted(crops.values(), key=lambda c: c["frame"]):
        by_fragment.setdefault(c["fragment"], []).append(c)
        by_identity.setdefault(c["identity"], []).append(c)
    ranked = sorted((r for r in rows if r["best"] is not None and np.isfinite(r["margin"])),
                    key=lambda r: -r["margin"])
    good = [r for r in ranked if r["correct"]][:n]
    bad = [r for r in ranked if not r["correct"]][:n]
    width = LABEL_W + 10 * 90
    lines = [_legend(width, [f"{fold} {tag}: left 5 = query fragment crops, right 5 = proposed identity's earlier crops",
                             "Sorted by margin. Labels are the mined (reviewed) number identities; masks not shown."])]
    for kind, group in (("CORRECT", good), ("WRONG", bad)):
        for r in group:
            q = by_fragment[r["fragment"]]
            g = [c for c in by_identity[r["best"]] if c["frame"] < q[0]["frame"]] or by_identity[r["best"]]
            pick = lambda xs: [xs[i] for i in np.linspace(0, len(xs) - 1, min(5, len(xs))).round().astype(int)]
            cells = [_fit(_crop(out, c)[0]) for c in pick(q)] + [_fit(_crop(out, c)[0]) for c in pick(g)]
            kind_ = kind if r["known"] else "WRONG (unknown player)" if kind == "WRONG" else kind
            lines.append(_row(cells, [kind_, f"query {r['identity'].split('|')[-2]} {r['identity'].split('|')[-1]}",
                                      f"proposed {r['best'].split('|')[-1]}",
                                      f"score {r['score']:.3f} margin {r['margin']:.3f}",
                                      r["fragment"].split(":", 1)[1]], width))
    path = out / "sheets" / f"examples_{fold}_{tag.replace('/', '_').replace('|', '_')}.jpg"
    cv2.imwrite(str(path), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path
