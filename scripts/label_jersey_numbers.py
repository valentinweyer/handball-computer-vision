"""Label jersey-number crops over an SSH-forwarded local web page.

The server binds to 127.0.0.1 by default. Labels autosave on the remote host,
so no browser downloads need to be copied back into the repository.

Examples:
    python -m scripts.label_jersey_numbers serve \
        --video data/raw/FelixClaar.mp4 \
        --samples runs/ocr_benchmark/FelixClaar_color_ab.json

    python -m scripts.label_jersey_numbers benchmark \
        --dataset runs/ocr_labels/FelixClaar/dataset.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import tempfile
from threading import Lock
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from handball_cv.jersey.identity import read_numbers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "runs/ocr_benchmark/FelixClaar_color_ab.json"
LABEL_RE = re.compile(r"^[0-9]{1,2}$")
VALID_STATUSES = {"readable", "unreadable", "unsure"}
BOX_STATUSES = {"complete", "partial", "too_loose", "not_number"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            # allow_nan=False: Python emits bare Infinity/NaN, which every strict
            # JSON reader rejects -- including the browser JSON.parse that consumes
            # these documents. Fail at write time rather than shipping a file that
            # only looks fine until something tries to read it.
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def load_source_samples(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("sample manifest must be a non-empty JSON list")
    seen = set()
    samples = []
    for position, item in enumerate(raw):
        index = int(item.get("index", position))
        if index in seen:
            raise ValueError(f"duplicate sample index: {index}")
        seen.add(index)
        box = item.get("box")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"sample {index} has no four-value box")
        samples.append({
            "index": index,
            "frame": int(item["frame"]),
            "box": [float(value) for value in box],
            "predictions": {
                key: str(item.get(key, "")).strip()
                for key in ("correct_bgr_input", "legacy_rgb_input")
                if key in item
            },
        })
    return samples


def clipped_box(box: list[float], width: int, height: int, pad: int = 0) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = np.rint(box).astype(int)
    values = (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )
    return tuple(int(value) for value in values)


# Project-wide class ids that count as "a person who can wear a jersey number".
# Referees (3) are deliberately excluded: a box on a referee is not a player number.
PLAYER_CLASS_IDS = (1, 2)


def max_player_containment(number_box, player_boxes) -> float:
    """Largest fraction of the number box's area covered by any single player box.

    Intersection-over-*number-area*, not IoU: a number is small and, when genuinely
    on a player, is fully inside that player's box, so this saturates at 1.0 while
    IoU would stay near zero regardless of how good the match is.
    """
    number = np.asarray(number_box, dtype=float)
    player_boxes = np.asarray(player_boxes, dtype=float).reshape(-1, 4)
    number_area = float(np.prod(np.maximum(number[2:] - number[:2], 0.0)))
    if number_area <= 0 or not len(player_boxes):
        return 0.0
    top_left = np.maximum(number[None, :2], player_boxes[:, :2])
    bottom_right = np.minimum(number[None, 2:], player_boxes[:, 2:])
    intersections = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=1)
    return float(intersections.max() / number_area)


def filter_samples_by_player_overlap(
    samples: list[dict], detections_path: Path, minimum_containment: float = 0.9
) -> list[dict]:
    """Keep number boxes substantially contained by a player prediction."""
    with np.load(detections_path, allow_pickle=False) as cache:
        offsets = np.asarray(cache["offsets"])
        boxes = np.asarray(cache["boxes"], dtype=float)
        class_ids = np.asarray(cache["class_id"])
    kept = []
    for sample in samples:
        frame = int(sample["frame"])
        if frame < 0 or frame + 1 >= len(offsets):
            raise ValueError(f"sample {sample['index']} frame is outside detection cache")
        start, end = int(offsets[frame]), int(offsets[frame + 1])
        player_boxes = boxes[start:end][
            np.isin(class_ids[start:end], PLAYER_CLASS_IDS)
        ]
        containment = max_player_containment(sample["box"], player_boxes)
        if containment >= minimum_containment:
            kept.append({**sample, "player_containment": containment})
    return kept


def build_dataset(
    video: Path,
    samples_path: Path,
    output_dir: Path,
    detections_path: Path,
    minimum_player_containment: float = 0.9,
) -> dict:
    video = video.resolve()
    samples_path = samples_path.resolve()
    output_dir = output_dir.resolve()
    crops_dir = output_dir / "crops"
    contexts_dir = output_dir / "contexts"
    crops_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir.mkdir(parents=True, exist_ok=True)
    detections_path = detections_path.resolve()
    all_source_samples = load_source_samples(samples_path)
    source_samples = filter_samples_by_player_overlap(
        all_source_samples, detections_path, minimum_player_containment
    )

    by_frame = defaultdict(list)
    for sample in source_samples:
        by_frame[sample["frame"]].append(sample)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video}")
    rendered = []
    try:
        for frame_index in sorted(by_frame):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not read frame {frame_index} from {video}")
            height, width = frame.shape[:2]
            for sample in by_frame[frame_index]:
                x1, y1, x2, y2 = clipped_box(sample["box"], width, height)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"sample {sample['index']} clips to an empty crop")
                crop = frame[y1:y2, x1:x2]
                crop_name = f"number_{sample['index']:04d}.jpg"
                if not cv2.imwrite(str(crops_dir / crop_name), crop):
                    raise RuntimeError(f"could not write crop {crop_name}")

                pad = max(24, round(max(x2 - x1, y2 - y1) * 0.8))
                cx1, cy1, cx2, cy2 = clipped_box(sample["box"], width, height, pad)
                context = frame[cy1:cy2, cx1:cx2].copy()
                cv2.rectangle(
                    context,
                    (x1 - cx1, y1 - cy1),
                    (x2 - cx1 - 1, y2 - cy1 - 1),
                    (0, 0, 255),
                    max(1, round(max(context.shape[:2]) / 180)),
                )
                context_name = f"number_{sample['index']:04d}.jpg"
                if not cv2.imwrite(str(contexts_dir / context_name), context):
                    raise RuntimeError(f"could not write context {context_name}")
                rendered.append({
                    **sample,
                    "sample_id": f"{video.stem}-f{frame_index:06d}-n{sample['index']:04d}",
                    "crop_path": f"crops/{crop_name}",
                    "context_path": f"contexts/{context_name}",
                    "crop_width": x2 - x1,
                    "crop_height": y2 - y1,
                })
    finally:
        capture.release()

    dataset = {
        "schema_version": 1,
        "video": str(video),
        "samples_source": str(samples_path),
        "player_detections": str(detections_path),
        "minimum_player_containment": minimum_player_containment,
        "source_sample_count": len(all_source_samples),
        "filtered_out_sample_count": len(all_source_samples) - len(source_samples),
        "created_at": utc_now(),
        "samples": sorted(rendered, key=lambda item: item["index"]),
    }
    write_json_atomic(output_dir / "dataset.json", dataset)
    return dataset


def validate_label(index: int, payload: dict, valid_indices: set[int]) -> dict:
    if index not in valid_indices:
        raise ValueError(f"unknown sample index: {index}")
    status = str(payload.get("status", ""))
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    value = payload.get("value")
    if status == "readable":
        value = str(value or "").strip()
        if not LABEL_RE.fullmatch(value):
            raise ValueError("readable labels must be one or two digits")
    else:
        value = None
    label = {"status": status, "value": value, "updated_at": utc_now()}
    box_status = payload.get("box_status")
    if box_status is not None:
        box_status = str(box_status)
        if box_status not in BOX_STATUSES:
            raise ValueError(f"invalid box_status: {box_status}")
        label["box_status"] = box_status
    return label


class LabelStore:
    def __init__(self, path: Path, dataset: dict, dataset_path: Path | None = None):
        self.path = path
        self.dataset = dataset
        self.dataset_path = (dataset_path or (path.parent / "dataset.json")).resolve()
        self.valid_indices = {int(sample["index"]) for sample in dataset["samples"]}
        self.lock = Lock()
        if path.is_file():
            raw = json.loads(path.read_text())
            all_labels = {
                **raw.get("excluded_labels", {}),
                **raw.get("labels", {}),
            }
            self.labels = {
                key: value for key, value in all_labels.items()
                if int(key) in self.valid_indices
            }
            self.excluded_labels = {
                key: value for key, value in all_labels.items()
                if int(key) not in self.valid_indices
            }
            write_json_atomic(self.path, self.document())
        else:
            self.labels = {}
            self.excluded_labels = {}

    def document(self) -> dict:
        return {
            "schema_version": 1,
            "video": self.dataset["video"],
            "dataset": str(self.dataset_path),
            "labels": self.labels,
            "excluded_labels": self.excluded_labels,
        }

    def save(self, index: int, payload: dict) -> dict:
        label = validate_label(index, payload, self.valid_indices)
        with self.lock:
            self.labels[str(index)] = label
            write_json_atomic(self.path, self.document())
        return label

    def clear(self, index: int) -> None:
        if index not in self.valid_indices:
            raise ValueError(f"unknown sample index: {index}")
        with self.lock:
            self.labels.pop(str(index), None)
            write_json_atomic(self.path, self.document())


LABEL_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jersey-number ground truth</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif}body{margin:0;background:#0d1117;color:#e6edf3}
header{position:sticky;top:0;z-index:2;padding:12px 18px;background:#0d1117f5;border-bottom:1px solid #30363d}
h1{font-size:20px;margin:0 0 5px}.sub,.help{color:#9da7b3;font-size:13px}main{max-width:1050px;margin:auto;padding:18px}
.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{background:#161b22;border:1px solid #30363d;border-radius:9px;padding:10px}
.panel span{display:block;color:#9da7b3;font-size:12px;margin-bottom:7px}img{display:block;width:100%;height:42vh;object-fit:contain;background:#05070a}
#crop{image-rendering:pixelated}.row{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;align-items:center}
input,button{font:inherit;border:1px solid #495362;background:#21262d;color:#e6edf3;border-radius:7px;padding:10px 12px}
input{font-size:26px;width:120px;text-align:center}button{cursor:pointer}button:hover{background:#30363d}.primary{background:#1f6f3d}
.danger{background:#6e3b24}.active{outline:2px solid #f2cc60}#meta{font-family:ui-monospace,monospace;color:#b9c2cc;margin:12px 0}
#prediction{display:none;padding:10px;border:1px dashed #495362;border-radius:7px;color:#f2cc60}.help{line-height:1.5;margin-top:14px}
@media(max-width:760px){.images{grid-template-columns:1fr}img{height:30vh}}
</style></head><body>
<header><h1>Jersey-number ground truth</h1><div class="sub" id="progress">Loading…</div></header>
<main><div class="images"><div class="panel"><span>Tight OCR crop</span><img id="crop"></div>
<div class="panel"><span>Context (red box is the labeled crop)</span><img id="context"></div></div>
<div id="meta"></div><div class="row"><input id="number" inputmode="numeric" maxlength="2" placeholder="0–99" autofocus>
<button class="primary" id="save">Save number ↵</button><button class="danger" id="unreadable">Unreadable U</button>
<button id="unsure">Unsure S</button><button id="clear">Clear</button></div>
<div class="row" id="box-status-row"><span class="sub" style="margin-right:4px">Box:</span>
<button data-box-status="complete" id="box-complete">Complete C</button>
<button data-box-status="partial" id="box-partial">Partial P</button>
<button data-box-status="too_loose" id="box-too-loose">Too loose L</button>
<button data-box-status="not_number" id="box-not-number">Not a number N</button></div>
<div class="row"><button id="previous">← Previous</button><button id="next">Next →</button>
<button id="next-unlabeled">Next unlabeled</button><button id="reveal">Reveal predictions</button></div>
<div id="prediction"></div><div class="help">Predictions are hidden to avoid anchoring your labels. Enter saves and advances.
Use U for unreadable, S for unsure, and arrow keys to navigate. C/P/L/N tag box geometry (complete/partial/too loose/not
a number) and travel with whichever readability verdict you save next. Every action autosaves to the remote machine.</div></main>
<script>
let dataset,labels,index=0,revealed=false,pendingBoxStatus=null;
const $=id=>document.getElementById(id);
async function request(path,options={}){const response=await fetch(path,options);const body=await response.json();if(!response.ok)throw new Error(body.error||response.statusText);return body}
function counts(){const values=Object.values(labels);return {done:values.length,readable:values.filter(x=>x.status==='readable').length,unreadable:values.filter(x=>x.status==='unreadable').length,unsure:values.filter(x=>x.status==='unsure').length}}
function renderBoxStatus(){document.querySelectorAll('#box-status-row button').forEach(button=>{button.classList.toggle('active',button.dataset.boxStatus===pendingBoxStatus)})}
function render(){const sample=dataset.samples[index],label=labels[String(sample.index)]||{};const c=counts();
$('progress').textContent=`${index+1} / ${dataset.samples.length} · ${c.done} labeled · ${c.readable} readable · ${c.unreadable} unreadable · ${c.unsure} unsure`;
$('crop').src='/'+sample.crop_path;$('context').src='/'+sample.context_path;
const provenance=sample.source_clip?` · ${sample.source_split||'?'}/${sample.source_clip}${sample.height_band?' · h-band '+sample.height_band:''}`:'';
$('meta').textContent=`${sample.sample_id} · frame ${sample.frame} · ${sample.crop_width}×${sample.crop_height}px · current: ${label.status||'unlabeled'} ${label.value??''}${provenance}`;
$('number').value=label.status==='readable'?(label.value||''):'';
pendingBoxStatus=label.box_status||null;renderBoxStatus();
$('prediction').textContent='Hidden model outputs: '+JSON.stringify(sample.predictions||{});$('prediction').style.display=revealed?'block':'none';
$('number').focus();$('number').select()}
function move(delta){index=Math.max(0,Math.min(dataset.samples.length-1,index+delta));render()}
function nextUnlabeled(){for(let offset=1;offset<=dataset.samples.length;offset++){const candidate=(index+offset)%dataset.samples.length;if(!labels[String(dataset.samples[candidate].index)]){index=candidate;render();return}}move(1)}
async function save(status){const sample=dataset.samples[index];const value=status==='readable'?$('number').value.trim():null;
const label=await request('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:sample.index,status,value,box_status:pendingBoxStatus})});labels[String(sample.index)]=label;nextUnlabeled()}
async function clearCurrent(){const sample=dataset.samples[index];await request('/api/clear',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:sample.index})});delete labels[String(sample.index)];render()}
$('save').onclick=()=>save('readable').catch(error=>alert(error.message));$('unreadable').onclick=()=>save('unreadable').catch(error=>alert(error.message));
$('unsure').onclick=()=>save('unsure').catch(error=>alert(error.message));$('clear').onclick=()=>clearCurrent().catch(error=>alert(error.message));
$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('next-unlabeled').onclick=nextUnlabeled;
$('reveal').onclick=()=>{revealed=!revealed;render()};
document.querySelectorAll('#box-status-row button').forEach(button=>{button.onclick=()=>{pendingBoxStatus=pendingBoxStatus===button.dataset.boxStatus?null:button.dataset.boxStatus;renderBoxStatus()}});
$('number').addEventListener('input',event=>{event.target.value=event.target.value.replace(/\D/g,'').slice(0,2)});
document.addEventListener('keydown',event=>{const key=event.key.toLowerCase();const boxKeys={c:'complete',p:'partial',l:'too_loose',n:'not_number'};
if(key in boxKeys){event.preventDefault();pendingBoxStatus=pendingBoxStatus===boxKeys[key]?null:boxKeys[key];renderBoxStatus()}
else if(key==='u'){event.preventDefault();save('unreadable').catch(error=>alert(error.message))}else if(key==='s'){event.preventDefault();save('unsure').catch(error=>alert(error.message))}else if(event.key==='ArrowLeft'){event.preventDefault();move(-1)}else if(event.key==='ArrowRight'){event.preventDefault();move(1)}else if(event.key==='Enter'){event.preventDefault();save('readable').catch(error=>alert(error.message))}});
request('/api/state').then(state=>{dataset=state.dataset;labels=state.labels;const first=dataset.samples.findIndex(sample=>!labels[String(sample.index)]);index=first<0?0:first;render()}).catch(error=>{$('progress').textContent=error.message});
</script></body></html>"""


def make_handler(output_dir: Path, dataset: dict, store: LabelStore):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, data: dict, status: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                body = LABEL_PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/state":
                self.send_json({"dataset": dataset, "labels": store.labels})
                return
            relative = Path(unquote(path.lstrip("/")))
            candidate = (output_dir / relative).resolve()
            if relative.parts and relative.parts[0] in {"crops", "contexts"} and candidate.is_relative_to(output_dir) and candidate.is_file():
                body = candidate.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_json({"error": "not found"}, 404)

        def read_payload(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                raise ValueError("invalid request size")
            return json.loads(self.rfile.read(length))

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self.read_payload()
                index = int(payload["index"])
                if self.path == "/api/label":
                    self.send_json(store.save(index, payload))
                elif self.path == "/api/clear":
                    store.clear(index)
                    self.send_json({"ok": True})
                else:
                    self.send_json({"error": "not found"}, 404)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, 400)

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def create_server(output_dir: Path, dataset: dict, store: LabelStore, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(output_dir, dataset, store))


def normalize_prediction(value) -> str:
    value = str(value or "").strip()
    return value if LABEL_RE.fullmatch(value) else ""


def score_model(
    readable_truth: dict[int, str],
    predictions: dict[int, str],
    unreadable_truth: set[int] | None = None,
) -> dict:
    unreadable_truth = unreadable_truth or set()
    confusion = defaultdict(Counter)
    correct = 0
    predicted = 0
    for index, truth in readable_truth.items():
        prediction = normalize_prediction(predictions.get(index, ""))
        confusion[truth][prediction or "<abstain>"] += 1
        predicted += bool(prediction)
        correct += prediction == truth
    total = len(readable_truth)
    unreadable_abstentions = sum(
        not normalize_prediction(predictions.get(index, ""))
        for index in unreadable_truth
    )
    return {
        "ground_truth_crops": total,
        "predicted_crops": predicted,
        "coverage": predicted / total if total else None,
        "accuracy": correct / total if total else None,
        "selective_accuracy": correct / predicted if predicted else None,
        "correct": correct,
        "unreadable_ground_truth_crops": len(unreadable_truth),
        "unreadable_abstentions": unreadable_abstentions,
        "unreadable_abstention_rate": (
            unreadable_abstentions / len(unreadable_truth)
            if unreadable_truth else None
        ),
        "confusion": {truth: dict(values) for truth, values in sorted(confusion.items())},
    }


def benchmark(dataset_path: Path, labels_path: Path, output_path: Path, device: str) -> dict:
    dataset = json.loads(dataset_path.read_text())
    labels = json.loads(labels_path.read_text()).get("labels", {})
    readable = {
        int(index): item["value"]
        for index, item in labels.items()
        if item.get("status") == "readable"
    }
    if not readable:
        raise ValueError("no readable ground-truth labels found")
    samples = {int(item["index"]): item for item in dataset["samples"]}
    predictions = {
        "smolvlm_color_corrected": {
            index: sample.get("predictions", {}).get("correct_bgr_input", "")
            for index, sample in samples.items()
        },
        "smolvlm_legacy_color": {
            index: sample.get("predictions", {}).get("legacy_rgb_input", "")
            for index, sample in samples.items()
        },
    }

    import easyocr
    reader = easyocr.Reader(["en"], gpu=device != "cpu", detector=False, verbose=False)
    easy_predictions = {}
    unreadable = {
        int(index) for index, item in labels.items()
        if item.get("status") == "unreadable"
    }
    for index in sorted(set(readable) | unreadable):
        sample = samples[index]
        crop_bgr = cv2.imread(str(dataset_path.parent / sample["crop_path"]))
        if crop_bgr is None:
            raise FileNotFoundError(sample["crop_path"])
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        height, width = crop_rgb.shape[:2]
        easy_predictions[index] = read_numbers(
            reader, crop_rgb, np.asarray([[0, 0, width, height]], dtype=float)
        )[0]
    predictions["easyocr_production"] = easy_predictions

    report = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "labels": str(labels_path.resolve()),
        "readable_labels": len(readable),
        "unreadable_labels": len(unreadable),
        "unsure_labels": sum(item.get("status") == "unsure" for item in labels.values()),
        "models": {
            name: score_model(readable, values, unreadable)
            for name, values in predictions.items()
        },
    }
    write_json_atomic(output_path, report)
    return report


def output_dir_for(video: Path, requested: Path | None) -> Path:
    return requested or ROOT / "runs/ocr_labels" / video.stem


def default_detection_cache(video: Path) -> Path:
    return ROOT / "outputs/team_comparison" / f".{video.stem}_detections_v1.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "serve"):
        command = subparsers.add_parser(name)
        command.add_argument("--video", required=True, type=Path)
        command.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
        command.add_argument(
            "--detections", type=Path,
            help="cached player predictions; defaults to outputs/team_comparison/.<video>_detections_v1.npz",
        )
        command.add_argument("--minimum-player-containment", type=float, default=0.9)
        command.add_argument("--output-dir", type=Path)
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8765)
    review = subparsers.add_parser(
        "review",
        help="serve a prebuilt dataset directory (e.g. from build_jersey_audit_set.py) "
        "without rebuilding it from a video",
    )
    review.add_argument("--dataset-dir", required=True, type=Path)
    review.add_argument(
        "--labels", type=Path,
        help="defaults to data/annotations/jersey/<dataset-dir name>_labels.json "
        "(tracked, unlike the generated dataset directory)",
    )
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8765)
    score = subparsers.add_parser("benchmark")
    score.add_argument("--dataset", required=True, type=Path)
    score.add_argument("--labels", type=Path)
    score.add_argument("--output", type=Path)
    score.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in {"build", "serve"}:
        output_dir = output_dir_for(args.video, args.output_dir)
        detections = args.detections or default_detection_cache(args.video)
        if not detections.is_file():
            raise FileNotFoundError(f"player detection cache not found: {detections}")
        dataset = build_dataset(
            args.video, args.samples, output_dir, detections,
            args.minimum_player_containment,
        )
        print(
            f"built {len(dataset['samples'])} player-overlapping crops "
            f"({dataset['filtered_out_sample_count']} filtered out) -> "
            f"{output_dir / 'dataset.json'}"
        )
        if args.command == "build":
            return
        store = LabelStore(output_dir / "labels.json", dataset)
        server = create_server(output_dir.resolve(), dataset, store, args.host, args.port)
        host, port = server.server_address
        print(f"labeler listening on http://{host}:{port}")
        print("Forward this port in VS Code's Ports panel, then open it in your local browser.")
        print(f"labels autosave to {store.path}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return

    if args.command == "review":
        output_dir = args.dataset_dir.resolve()
        dataset_path = output_dir / "dataset.json"
        dataset = json.loads(dataset_path.read_text())
        labels_path = (
            args.labels
            or ROOT / "data/annotations/jersey" / f"{output_dir.name}_labels.json"
        ).resolve()
        store = LabelStore(labels_path, dataset, dataset_path=dataset_path)
        server = create_server(output_dir, dataset, store, args.host, args.port)
        host, port = server.server_address
        print(f"reviewer listening on http://{host}:{port}")
        print("Forward this port in VS Code's Ports panel, then open it in your local browser.")
        print(f"labels autosave to {store.path}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return

    dataset_path = args.dataset.resolve()
    labels_path = (args.labels or dataset_path.parent / "labels.json").resolve()
    output_path = (args.output or dataset_path.parent / "benchmark.json").resolve()
    report = benchmark(dataset_path, labels_path, output_path, args.device)
    print(json.dumps(report, indent=2))
    print(f"benchmark written to {output_path}")


if __name__ == "__main__":
    main()
