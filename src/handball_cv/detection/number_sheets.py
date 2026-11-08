"""Jersey numbers from player sheets: one detector pass per frame instead of one per player.

Every player box of a frame is cut out exactly (no margin), scaled to one tile
height and packed in rows into a 704x704 sheet; a one-class RF-DETR trained on
that layout finds the numbers, and each is mapped back to the frame through the
tile holding its centre. The packing has to be exactly what the model was
trained on.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

SIZE = 704
GAP = 4
FILL = (114, 114, 114)
TILE_HEIGHT = 224       # one sheet per frame
HOST_COVERAGE = 0.5     # share of a number box that must lie on its player's box
NMS_IOU = 0.5           # one number found from two overlapping tiles is kept once
NUMBER_CLASS_ID = 4


def tile_size(width: float, height: float, h: int) -> tuple[int, int]:
    """(w, h) of a width x height source box drawn at tile height `h`.

    A tile keeps its aspect; one wider than tall is fitted into h x h instead.
    """
    if width > height:
        return h, max(1, round(h * height / width))
    return max(1, round(h * width / height)), h


def pack_rows(sizes: list[tuple[int, int]], h: int) -> list[tuple[int, int, int]]:
    """(sheet, x, y) per tile, in the given order.

    Rows of height `h` run left to right with GAP between tiles and rows; a tile
    that does not fit starts the next row, a row that does not fit the next sheet.
    """
    place, sheet, x, y = [], 0, 0, 0
    for w, _ in sizes:
        if x and x + w > SIZE:
            x, y = 0, y + h + GAP
        if y + h > SIZE:
            sheet, x, y = sheet + 1, 0, 0
        place.append((sheet, x, y))
        x += w + GAP
    return place


def render(crops: list[Image.Image], h: int, aspects: list[float] | None = None):
    """Sheets for `crops` at tile height `h`, plus each tile's (sheet, x, y, w, h).

    `aspects` overrides a crop's own width/height ratio (the dataset's stretched
    640x640 frames are drawn at their 16:9 source geometry).
    """
    sizes = [tile_size(a * c.height if aspects else c.width, c.height, h)
             for c, a in zip(crops, aspects or [None] * len(crops))]
    place = pack_rows(sizes, h)
    sheets = [Image.new("RGB", (SIZE, SIZE), FILL) for _ in range(max(p[0] for p in place) + 1)] if crops else []
    tiles = []
    for crop, (w, th), (k, x, y) in zip(crops, sizes, place):
        sheets[k].paste(crop.resize((w, th), Image.BILINEAR), (x, y))
        tiles.append((k, x, y, w, th))
    return sheets, tiles


def host_crops(image: Image.Image, hosts_xyxy: np.ndarray) -> tuple[list, list]:
    """Each player box cut out exactly: (crops, (bounds, host_xyxy) per crop)."""
    crops, context = [], []
    for x1, y1, x2, y2 in np.asarray(hosts_xyxy, dtype=float).reshape(-1, 4):
        bounds = (max(0, int(x1)), max(0, int(y1)),
                  min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2)))
        if bounds[2] - bounds[0] < 2 or bounds[3] - bounds[1] < 2:
            continue
        crops.append(image.crop(bounds))
        context.append((bounds, (x1, y1, x2, y2)))
    return crops, context


def _on_host(box: tuple, host: tuple) -> bool:
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return False
    w = max(0.0, min(x2, host[2]) - max(x1, host[0]))
    h = max(0.0, min(y2, host[3]) - max(y1, host[1]))
    return w * h / area >= HOST_COVERAGE


def _iou(a: tuple, b: tuple) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def deduplicate(boxes: list[tuple], scores: list[float]) -> tuple[list, list]:
    """Greedy NMS, highest score first; survivors in (x, y, w, h) order (matching
    depends on row order).
    """
    kept = []
    for i in sorted(range(len(boxes)), key=lambda i: -scores[i]):
        if all(_iou(boxes[i], boxes[j]) < NMS_IOU for j in kept):
            kept.append(i)
    kept.sort(key=lambda i: (boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]))
    return [boxes[i] for i in kept], [scores[i] for i in kept]


def from_sheets(detections: list, crops: list, context: list, tiles: list) -> tuple[list, list]:
    """Sheet detections back to frame boxes, kept when on their own player.

    `detections[k]` is `(xyxy, scores)` for sheet k. A box goes to the tile
    holding its centre, scaled by that tile's size against its crop's.
    """
    boxes, scores = [], []
    for k, (xyxy, conf) in enumerate(detections):
        for (x1, y1, x2, y2), score in zip(xyxy, conf):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            for crop, (bounds, host), (kk, tx, ty, tw, th) in zip(crops, context, tiles):
                if kk == k and tx <= cx < tx + tw and ty <= cy < ty + th:
                    sx, sy = tw / crop.width, th / crop.height
                    box = (bounds[0] + (x1 - tx) / sx, bounds[1] + (y1 - ty) / sy,
                           bounds[0] + (x2 - tx) / sx, bounds[1] + (y2 - ty) / sy)
                    if _on_host(box, host):
                        boxes.append(box)
                        scores.append(float(score))
                    break
    return boxes, scores


class SheetNumberDetector:
    """`detect(frame_rgb, hosts_xyxy, threshold) -> (xyxy, scores)` from a TorchScript export."""

    def __init__(self, weights, device: str = "cuda", tile_height: int = TILE_HEIGHT) -> None:
        from handball_cv.detection.rfdetr_torchscript import TorchScriptDetector

        self.model = TorchScriptDetector(weights, device=device, resolution=SIZE,
                                         class_map={0: NUMBER_CLASS_ID})
        self.tile_height = tile_height

    def detect(self, frame_rgb: np.ndarray, hosts_xyxy: np.ndarray, threshold: float):
        image = Image.fromarray(frame_rgb)
        crops, context = host_crops(image, hosts_xyxy)
        if not crops:
            return np.zeros((0, 4)), np.zeros(0)
        sheets, tiles = render(crops, self.tile_height)
        detections = [self.model.infer(np.asarray(sheet), threshold)[:2] for sheet in sheets]
        boxes, scores = deduplicate(*from_sheets(detections, crops, context, tiles))
        return np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(scores, dtype=float)


def replace_number_rows(cache: dict, frame_rgb, detector: SheetNumberDetector, *, frames=None,
                        host_classes=(1, 2), detector_id: str = "") -> dict:
    """`cache` with its number rows replaced by sheet detections on its own player boxes.

    `frame_rgb(index)` decodes a frame; it is called only for frames the cache
    holds a player on (with a detection stride, the scored ones). `frames`
    iterates the cache's frame indices in order (pass a progress bar over
    them); by default all of them. Every other row is copied unchanged, so the
    result stands in for the cache everywhere, and the threshold is the cache's.
    """
    offsets, boxes, conf, cls = cache["offsets"], cache["boxes"], cache["confidence"], cache["class_id"]
    threshold = float(cache["threshold"])
    out_offsets, out_boxes, out_conf, out_cls = [0], [], [], []
    for frame in frames if frames is not None else range(len(offsets) - 1):
        lo, hi = int(offsets[frame]), int(offsets[frame + 1])
        keep = [i for i in range(lo, hi) if cls[i] != NUMBER_CLASS_ID]
        hosts = boxes[[i for i in keep if cls[i] in host_classes]]
        numbers, scores = (detector.detect(frame_rgb(frame), hosts, threshold) if len(hosts)
                           else (np.zeros((0, 4)), np.zeros(0)))
        out_boxes += [boxes[i] for i in keep] + list(numbers)
        out_conf += [conf[i] for i in keep] + list(scores)
        out_cls += [cls[i] for i in keep] + [NUMBER_CLASS_ID] * len(numbers)
        out_offsets.append(len(out_boxes))
    if len(out_offsets) != len(offsets):
        raise ValueError(f"frames covered {len(out_offsets) - 1} of the cache's {len(offsets) - 1}")
    return {**cache,
            "detector_id": np.asarray(f"{cache['detector_id']} + sheet numbers {detector_id}"),
            "offsets": np.asarray(out_offsets, dtype=np.int64),
            "boxes": np.asarray(out_boxes, dtype=float).reshape(-1, 4),
            "confidence": np.asarray(out_conf, dtype=float),
            "class_id": np.asarray(out_cls, dtype=np.int64)}
