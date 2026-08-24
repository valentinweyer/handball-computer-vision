"""Build and ingest an easy, tracker-independent team-labeling page.

The generated page works as a local file in a browser.  Labels are persisted
in browser localStorage while working; the Export button downloads a validated
manifest which this script can ingest back into the repository.

Examples:
    conda run -n NewEnv python notebooks/label_team_detections.py build \
        FelixClaar.mp4 Hannover.mp4 --frames-per-video 12

    conda run -n NewEnv python notebooks/label_team_detections.py ingest \
        ~/Downloads/FelixClaar-team-labels.json \
        --output annotations/team/FelixClaar.json
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from team_dataset import (
    annotation_counts,
    annotation_from_code,
    build_sample_records,
    evenly_spaced_detection_frames,
    load_detection_cache,
    new_manifest,
    read_manifest,
    render_sample_assets,
    validate_manifest,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "outputs" / "team_comparison"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "team_dataset"
DEFAULT_ANNOTATION_DIR = ROOT / "annotations" / "team"
DEFAULT_CLASS_IDS = {1, 2}


def _box_iou(box: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if not len(candidates):
        return np.empty(0, dtype=float)
    top_left = np.maximum(box[None, :2], candidates[:, :2])
    bottom_right = np.minimum(box[None, 2:], candidates[:, 2:])
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    box_area = max(float(np.prod(np.maximum(box[2:] - box[:2], 0.0))), 0.0)
    candidate_area = np.prod(
        np.maximum(candidates[:, 2:] - candidates[:, :2], 0.0), axis=1
    )
    union = box_area + candidate_area - intersection
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )


def legacy_frame_indices(labels_path: Path) -> set[int]:
    raw = json.loads(labels_path.read_text()).get("labels", {})
    return {int(key.split("_", 1)[0]) for key in raw}


def migrate_legacy_labels(
    samples: list[dict],
    cache: dict,
    run_path: Path,
    labels_path: Path,
    frame_offset: int = 0,
    minimum_iou: float = 0.85,
) -> dict[str, int]:
    """Reuse manually reviewed displayed crops by matching their exact boxes."""
    with np.load(run_path, allow_pickle=True) as raw:
        run_boxes = raw["boxes"]
    labels = json.loads(labels_path.read_text()).get("labels", {})
    sample_lookup = {
        (int(sample["frame_index"]), int(sample["detection_index"])): sample
        for sample in samples
    }
    offsets = np.asarray(cache["offsets"])
    cache_boxes = np.asarray(cache["boxes"])
    label_to_code = {
        "TEAM A": "A", "TEAM B": "B", "A": "A", "B": "B",
        "GK": "G", "GOALKEEPER": "G", "REFEREE": "R", "OTHER": "O",
        "MIXED": "M", "UNCLEAR": "X", "SKIP": "X",
    }
    counts = {"migrated": 0, "unmatched": 0, "ambiguous": 0}
    for key, label in labels.items():
        row_text, column_text = key.split("_", 1)
        row_index, column = int(row_text), int(column_text)
        frame_index = row_index + frame_offset
        if row_index >= len(run_boxes) or column >= run_boxes.shape[1]:
            counts["unmatched"] += 1
            continue
        box = np.asarray(run_boxes[row_index, column], dtype=float)
        if not np.isfinite(box).all() or frame_index < 0 or frame_index + 1 >= len(offsets):
            counts["unmatched"] += 1
            continue
        start, end = int(offsets[frame_index]), int(offsets[frame_index + 1])
        overlaps = _box_iou(box, cache_boxes[start:end])
        if not len(overlaps) or float(overlaps.max()) < minimum_iou:
            counts["unmatched"] += 1
            continue
        winners = np.flatnonzero(np.isclose(overlaps, overlaps.max(), atol=1e-6))
        if len(winners) != 1:
            counts["ambiguous"] += 1
            continue
        detection_index = int(winners[0])
        sample = sample_lookup.get((frame_index, detection_index))
        code = label_to_code.get(str(label).upper())
        if sample is None or code is None:
            counts["unmatched"] += 1
            continue
        sample["annotation"] = annotation_from_code(code)
        sample["annotation_source"] = {
            "kind": "legacy_reviewed_crop",
            "run": str(run_path.resolve()),
            "labels": str(labels_path.resolve()),
            "iou": float(overlaps[detection_index]),
        }
        counts["migrated"] += 1
    return counts


def preserve_existing_annotations(samples: list[dict], old_manifest: dict) -> int:
    old_by_id = {item["sample_id"]: item for item in old_manifest["samples"]}
    restored = 0
    for sample in samples:
        previous = old_by_id.get(sample["sample_id"])
        if previous and previous.get("annotation", {}).get("code") is not None:
            sample["annotation"] = previous["annotation"]
            sample["annotation_source"] = previous.get(
                "annotation_source", {"kind": "previous_manifest"}
            )
            restored += 1
    return restored


def write_label_page(path: Path, manifest: dict) -> None:
    serialized = json.dumps(manifest, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(f"{manifest['video_id']} team labels")
    storage_key = html.escape(
        f"team-labels-v{manifest['schema_version']}-{manifest['video_id']}-"
        f"{manifest['video_size']}-{manifest['video_mtime_ns']}"
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
  body {{ margin: 0; background: #0d1117; color: #e6edf3; }}
  header {{ position: sticky; top: 0; z-index: 2; padding: 12px 18px;
    background: rgba(13,17,23,.96); border-bottom: 1px solid #30363d; }}
  h1 {{ margin: 0 0 5px; font-size: 20px; }}
  .sub {{ color: #9da7b3; font-size: 13px; }}
  main {{ max-width: 1120px; margin: 0 auto; padding: 18px; }}
  #preview {{ display: block; width: 100%; max-height: 62vh; object-fit: contain;
    background: #161b22; border: 1px solid #30363d; border-radius: 9px; }}
  #meta {{ margin: 10px 0; font-family: ui-monospace, monospace; color: #b9c2cc; }}
  .controls {{ display: grid; grid-template-columns: repeat(7, minmax(90px, 1fr)); gap: 8px; }}
  button, .file-label {{ border: 1px solid #495362; background: #21262d; color: #e6edf3;
    border-radius: 7px; padding: 10px 8px; cursor: pointer; text-align: center; }}
  button:hover, .file-label:hover {{ background: #30363d; }}
  button strong {{ color: #7ee787; }}
  .quality, .actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
  .active {{ outline: 2px solid #f2cc60; background: #3b321d; }}
  #status {{ margin: 13px 0 8px; font-weight: 600; }}
  #current {{ color: #f2cc60; }}
  input[type=file] {{ display: none; }}
  .help {{ margin-top: 14px; color: #9da7b3; line-height: 1.5; }}
  @media (max-width: 760px) {{ .controls {{ grid-template-columns: repeat(3, 1fr); }} }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">A/B are anonymous inside this video only. Each image is one RF-DETR detection—no tracker labels.</div>
</header>
<main>
  <div id="status"></div>
  <img id="preview" alt="player detection preview">
  <div id="meta"></div>
  <div class="controls" id="semantic-controls">
    <button data-code="A"><strong>A</strong> · Team A</button>
    <button data-code="B"><strong>B</strong> · Team B</button>
    <button data-code="G"><strong>G</strong> · Goalkeeper</button>
    <button data-code="R"><strong>R</strong> · Referee</button>
    <button data-code="O"><strong>O</strong> · Other</button>
    <button data-code="M"><strong>M</strong> · Mixed</button>
    <button data-code="X"><strong>X</strong> · Unusable</button>
  </div>
  <div class="quality" id="quality-controls">
    <button data-quality="clean"><strong>1</strong> clean</button>
    <button data-quality="occluded"><strong>2</strong> occluded but readable</button>
    <button data-quality="mixed"><strong>3</strong> mixed people</button>
    <button data-quality="unusable"><strong>4</strong> unusable</button>
  </div>
  <div class="actions">
    <button id="previous">← previous</button>
    <button id="next">next →</button>
    <button id="unlabeled">next unlabeled</button>
    <button id="clear">clear current</button>
    <button id="export">export JSON</button>
    <label class="file-label">import JSON<input id="import" type="file" accept="application/json"></label>
  </div>
  <div class="help">
    Fast path: press A/B/G/R/O/M/X and the page advances automatically. For a readable but occluded
    player, press 2 before A or B. Arrow keys navigate. Progress is saved in this browser; Export JSON
    when finished. Mixed means another person's kit contaminates the target crop. Unusable means the
    intended player or jersey evidence is effectively absent.
  </div>
</main>
<script>
const original = {serialized};
const storageKey = {json.dumps(storage_key)};
let manifest = structuredClone(original);
let index = 0;
let pendingQuality = "clean";

const codeDefaults = {{
  A: {{team:"A", role:"field", quality:"clean"}},
  B: {{team:"B", role:"field", quality:"clean"}},
  G: {{team:null, role:"goalkeeper", quality:"clean"}},
  R: {{team:null, role:"referee", quality:"clean"}},
  O: {{team:null, role:"other", quality:"clean"}},
  M: {{team:null, role:"unknown", quality:"mixed"}},
  X: {{team:null, role:"unknown", quality:"unusable"}}
}};

function savedAnnotations() {{
  try {{ return JSON.parse(localStorage.getItem(storageKey) || "{{}}"); }}
  catch (_) {{ return {{}}; }}
}}
const saved = savedAnnotations();
for (const sample of manifest.samples) {{
  if (saved[sample.sample_id]) sample.annotation = saved[sample.sample_id];
}}

function persist() {{
  const annotations = {{}};
  for (const sample of manifest.samples) annotations[sample.sample_id] = sample.annotation;
  localStorage.setItem(storageKey, JSON.stringify(annotations));
}}

function labeledCount() {{
  return manifest.samples.filter(sample => sample.annotation && sample.annotation.code).length;
}}

function render() {{
  const sample = manifest.samples[index];
  document.getElementById("preview").src = sample.preview_path;
  const annotation = sample.annotation || {{code:null}};
  document.getElementById("status").innerHTML =
    `<span id="current">${{index + 1}} / ${{manifest.samples.length}}</span> · ` +
    `${{labeledCount()}} labeled · current: ${{annotation.code || "—"}}`;
  const q = sample.crop_quality || {{}};
  document.getElementById("meta").textContent =
    `${{sample.sample_id}} | box ${{Math.round(sample.bbox_width)}}×${{Math.round(sample.bbox_height)}} ` +
    `| detector ${{sample.detector_confidence.toFixed(2)}} | geometric overlap ` +
    `${{sample.torso_contamination.toFixed(2)}} | quality ${{(q.score || 0).toFixed(2)}}`;
  document.querySelectorAll("[data-code]").forEach(button =>
    button.classList.toggle("active", button.dataset.code === annotation.code));
  document.querySelectorAll("[data-quality]").forEach(button =>
    button.classList.toggle("active", button.dataset.quality === pendingQuality));
}}

function move(delta) {{
  index = Math.max(0, Math.min(manifest.samples.length - 1, index + delta));
  const currentQuality = manifest.samples[index].annotation?.quality;
  pendingQuality = currentQuality || "clean";
  render();
}}

function nextUnlabeled() {{
  const count = manifest.samples.length;
  for (let step = 1; step <= count; step++) {{
    const candidate = (index + step) % count;
    if (!manifest.samples[candidate].annotation?.code) {{
      index = candidate; pendingQuality = "clean"; render(); return;
    }}
  }}
  render();
}}

function annotate(code) {{
  const base = structuredClone(codeDefaults[code]);
  if (["A","B","G","R","O"].includes(code)) base.quality = pendingQuality;
  manifest.samples[index].annotation = {{code, ...base}};
  persist();
  pendingQuality = "clean";
  nextUnlabeled();
}}

function setQuality(quality) {{ pendingQuality = quality; render(); }}
for (const button of document.querySelectorAll("[data-code]"))
  button.addEventListener("click", () => annotate(button.dataset.code));
for (const button of document.querySelectorAll("[data-quality]"))
  button.addEventListener("click", () => setQuality(button.dataset.quality));
document.getElementById("previous").onclick = () => move(-1);
document.getElementById("next").onclick = () => move(1);
document.getElementById("unlabeled").onclick = nextUnlabeled;
document.getElementById("clear").onclick = () => {{
  manifest.samples[index].annotation = {{code:null,team:null,role:null,quality:null}};
  persist(); render();
}};
document.getElementById("export").onclick = () => {{
  const blob = new Blob([JSON.stringify(manifest, null, 2) + "\\n"], {{type:"application/json"}});
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `${{manifest.video_id}}-team-labels.json`;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}};
document.getElementById("import").onchange = async event => {{
  const incoming = JSON.parse(await event.target.files[0].text());
  if (incoming.video_id !== manifest.video_id || incoming.samples.length !== manifest.samples.length) {{
    alert("That export belongs to a different dataset build."); return;
  }}
  const incomingById = Object.fromEntries(incoming.samples.map(sample => [sample.sample_id, sample]));
  for (const sample of manifest.samples) {{
    if (incomingById[sample.sample_id]) sample.annotation = incomingById[sample.sample_id].annotation;
  }}
  persist(); render();
}};

document.addEventListener("keydown", event => {{
  if (event.target.tagName === "INPUT") return;
  const key = event.key.toUpperCase();
  if (codeDefaults[key]) {{ event.preventDefault(); annotate(key); return; }}
  if (event.key === "ArrowLeft") {{ event.preventDefault(); move(-1); }}
  if (event.key === "ArrowRight") {{ event.preventDefault(); move(1); }}
  if (["1","2","3","4"].includes(event.key)) {{
    setQuality({{"1":"clean","2":"occluded","3":"mixed","4":"unusable"}}[event.key]);
  }}
}});

const firstUnlabeled = manifest.samples.findIndex(sample => !sample.annotation?.code);
if (firstUnlabeled >= 0) index = firstUnlabeled;
render();
</script>
</body>
</html>
"""
    path.write_text(page)


def command_build(args: argparse.Namespace) -> None:
    if (args.legacy_run is None) != (args.legacy_labels is None):
        raise SystemExit("--legacy-run and --legacy-labels must be provided together")
    if args.legacy_run is not None and len(args.videos) != 1:
        raise SystemExit("legacy migration currently accepts exactly one video")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video_argument in args.videos:
        video = video_argument.resolve()
        cache_path = args.cache_dir / f".{video.stem}_detections_v1.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"missing RF-DETR cache {cache_path}; run render_team_comparison.py "
                "once to create detections"
            )
        cache = load_detection_cache(cache_path)
        base_frames = set(
            evenly_spaced_detection_frames(cache, args.frames_per_video)
        )
        frames = set(base_frames)
        if args.legacy_labels is not None:
            frames |= {
                frame + args.legacy_frame_offset
                for frame in legacy_frame_indices(args.legacy_labels)
            }
        samples = build_sample_records(
            video,
            cache,
            frames,
            class_ids=set(args.class_ids),
            max_per_frame=args.max_per_frame,
        )
        dataset_dir = args.output_dir / video.stem
        manifest_path = dataset_dir / "manifest.json"
        old_manifest = None
        if manifest_path.exists():
            old_manifest = read_manifest(manifest_path)
        restored = preserve_existing_annotations(samples, old_manifest) if old_manifest else 0
        migration = None
        if args.legacy_run is not None:
            migration = migrate_legacy_labels(
                samples,
                cache,
                args.legacy_run,
                args.legacy_labels,
                frame_offset=args.legacy_frame_offset,
                minimum_iou=args.legacy_minimum_iou,
            )
            # Legacy labels can span many source frames. Keep only the exact
            # reviewed detections from those extra frames; including every
            # nearby detection would turn reuse into hundreds of new labels.
            samples = [
                sample for sample in samples
                if int(sample["frame_index"]) in base_frames
                or sample["annotation"].get("code") is not None
            ]
        render_sample_assets(video, samples, dataset_dir)
        manifest = new_manifest(video, cache_path, cache, samples)
        write_manifest(manifest_path, manifest)
        page_path = dataset_dir / "label.html"
        write_label_page(page_path, manifest)
        print(f"{video.name}: {len(samples)} detector crops")
        if restored:
            print(f"  restored {restored} labels from the previous manifest")
        if migration is not None:
            print(f"  legacy migration: {migration}")
        print(f"  labels: {annotation_counts(manifest)}")
        print(f"  open: {page_path}")


def command_ingest(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.export)
    output = args.output
    if output is None:
        output = DEFAULT_ANNOTATION_DIR / f"{manifest['video_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(output, manifest)
    print(f"wrote {output}")
    print(f"counts: {annotation_counts(manifest)}")


def command_summary(args: argparse.Namespace) -> None:
    for path in args.manifests:
        manifest = read_manifest(path)
        print(f"{path}: {annotation_counts(manifest)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("videos", nargs="+", type=Path)
    build.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build.add_argument("--frames-per-video", type=int, default=12)
    build.add_argument("--max-per-frame", type=int, default=14)
    build.add_argument("--class-ids", type=int, nargs="+", default=sorted(DEFAULT_CLASS_IDS))
    build.add_argument("--legacy-run", type=Path)
    build.add_argument("--legacy-labels", type=Path)
    build.add_argument("--legacy-frame-offset", type=int, default=0)
    build.add_argument("--legacy-minimum-iou", type=float, default=0.85)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("export", type=Path)
    ingest.add_argument("--output", type=Path)
    summary = commands.add_parser("summary")
    summary.add_argument("manifests", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        command_build(args)
    elif args.command == "ingest":
        command_ingest(args)
    else:
        command_summary(args)


if __name__ == "__main__":
    main()
