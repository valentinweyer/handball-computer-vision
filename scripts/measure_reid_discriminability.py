"""Can the re-ID embedding tell two teammates apart?

`PlayerRegistry.reid_match` decides who a reappearing player is from one frozen
SigLIP embedding, thresholded at `REID_COS_SIM_MIN`, with only team and
goalkeeper role available as vetoes. Within a team those vetoes are silent: every
outfield player wears the same kit, so the embedding is the *entire*
discriminator. Whether it carries enough signal at production crop sizes has
never been measured -- this measures it.

Ground truth comes from the labelled 1080p evaluation set: two readable crops in
the same clip carrying the same jersey number are the same person, and two
carrying different numbers are not. Each labelled number box is traced back to
the player box containing it (the same containment rule the set was built with),
and that player box is embedded exactly as the runtime embeds it.

Reported:
  - similarity of same-person pairs vs different-person pairs on the same team
  - how many different-person pairs clear REID_COS_SIM_MIN, i.e. are eligible
    to be matched to the wrong player
  - rank-1 retrieval: nearest neighbour in the same clip -- the actual operation
    `reid_match` performs -- and how often it lands on the right person, against
    the chance floor, since a clip with few players is easy for any embedding

`--embedding` picks what is being judged: `siglip` is the team model's feature
extractor, which is what `reid_match` uses today; `prtreid` is the SoccerNet
person-reID checkpoint already in `models/`, which nothing currently consumes.
The crop follows the embedding -- SigLIP sees the torso box the team model
defines, PRTReID the whole person box it was trained on.

Example:
    python -m scripts.measure_reid_discriminability \
        --dataset runs/number_eval_1080p/dataset.json \
        --labels data/annotations/jersey/number_eval_1080p_labels.json \
        --team-model outputs/team_dataset/.Melsungen_window_team.pkl
    python -m scripts.measure_reid_discriminability \
        --dataset runs/number_eval_1080p/dataset.json \
        --labels data/annotations/jersey/number_eval_1080p_labels.json \
        --embedding prtreid
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from handball_cv.teams.model import TeamModel, torso_boxes
from handball_cv.tracking.identity import REID_COS_SIM_MIN
from scripts.label_jersey_numbers import PLAYER_CLASS_IDS, write_json_atomic
from scripts.render_raw_team_classification import frame_detections


def containing_player_box(number_box, cache, frame_index):
    """The player box that best contains this number box, or None.

    Intersection-over-number-area, matching `max_player_containment`: the number
    is small and sits wholly inside its wearer's box, so this saturates at 1.0.
    """
    detections = frame_detections(cache, frame_index)
    boxes = detections.xyxy[np.isin(detections.class_id, PLAYER_CLASS_IDS)]
    number = np.asarray(number_box, dtype=float)
    area = float(np.prod(np.maximum(number[2:] - number[:2], 0.0)))
    if area <= 0 or not len(boxes):
        return None
    top_left = np.maximum(number[None, :2], boxes[:, :2])
    bottom_right = np.minimum(number[None, 2:], boxes[:, 2:])
    overlap = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=1) / area
    best = int(np.argmax(overlap))
    return boxes[best] if overlap[best] >= 0.9 else None


def collect_crops(samples, video_path: Path, cache_path: Path, full_box=False):
    """Torso crops for each labelled sample, in the runtime's crop geometry."""
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {k: np.asarray(handle[k]) for k in handle.files}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    crops, kept = [], []
    for sample in sorted(samples, key=lambda s: s["frame"]):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame"]))
        ok, frame_bgr = capture.read()
        if not ok:
            continue
        player_box = containing_player_box(sample["box"], cache, int(sample["frame"]))
        if player_box is None:
            continue
        box = player_box[None, :] if full_box else torso_boxes(player_box[None, :])
        x1, y1, x2, y2 = box[0].round().astype(int)
        height, width = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crops.append(cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        kept.append({**sample, "player_box": player_box.tolist()})
    capture.release()
    return crops, kept


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    unit = embeddings / norms
    return unit @ unit.T


def analyse_clip(similarity, numbers, teams) -> dict:
    """Pair statistics and rank-1 retrieval for one clip."""
    count = len(numbers)
    same_person, other_same_team, other_any = [], [], []
    for i in range(count):
        for j in range(i + 1, count):
            value = float(similarity[i, j])
            if numbers[i] == numbers[j]:
                same_person.append(value)
            else:
                other_any.append(value)
                if teams[i] == teams[j]:
                    other_same_team.append(value)

    correct = considered = correct_same_team = considered_same_team = 0
    chance = chance_same_team = 0.0
    for i in range(count):
        gallery = [j for j in range(count) if j != i]
        if not gallery or numbers.count(numbers[i]) < 2:
            continue          # no correct answer exists for this query
        considered += 1
        nearest = max(gallery, key=lambda j: similarity[i, j])
        correct += numbers[nearest] == numbers[i]
        # A gallery of k candidates of whom m are the right person answers
        # correctly m/k of the time by guessing. Rank-1 only means something
        # against that, and a clip with few players has a high floor.
        chance += sum(numbers[j] == numbers[i] for j in gallery) / len(gallery)
        same_team = [j for j in gallery if teams[j] == teams[i]]
        if same_team and any(numbers[j] == numbers[i] for j in same_team):
            considered_same_team += 1
            nearest = max(same_team, key=lambda j: similarity[i, j])
            correct_same_team += numbers[nearest] == numbers[i]
            chance_same_team += (
                sum(numbers[j] == numbers[i] for j in same_team) / len(same_team)
            )

    def describe(values):
        if not values:
            return None
        array = np.asarray(values)
        return {
            "n": int(array.size),
            "mean": float(array.mean()),
            "p10": float(np.percentile(array, 10)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
            "max": float(array.max()),
            "frac_above_reid_floor": float((array >= REID_COS_SIM_MIN).mean()),
        }

    return {
        "crops": count,
        "distinct_numbers": len(set(numbers)),
        "same_person_pairs": describe(same_person),
        "different_person_same_team_pairs": describe(other_same_team),
        "different_person_any_team_pairs": describe(other_any),
        "rank1_any_team": {
            "queries": considered,
            "correct": correct,
            "accuracy": correct / considered if considered else None,
            "chance": chance / considered if considered else None,
        },
        "rank1_same_team_gallery": {
            "queries": considered_same_team,
            "correct": correct_same_team,
            "accuracy": (
                correct_same_team / considered_same_team
                if considered_same_team else None
            ),
            "chance": (
                chance_same_team / considered_same_team
                if considered_same_team else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--team-model", type=Path,
                        help="only its SigLIP feature extractor is used; the "
                             "per-video team classifier is not")
    parser.add_argument(
        "--embedding", default="siglip", choices=("siglip", "prtreid"),
        help="siglip is what reid_match uses today; prtreid is the person "
             "re-identification checkpoint already in models/",
    )
    parser.add_argument("--prtreid-root", type=Path, default=Path("prtreid-upstream"))
    parser.add_argument(
        "--prtreid-checkpoint", type=Path,
        default=Path("models/prtreid/prtreid-soccernet-baseline.pth.tar"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text())
    labels = json.loads(args.labels.read_text())["labels"]
    readable = {
        int(index): entry["value"] for index, entry in labels.items()
        if entry.get("status") == "readable" and entry.get("value")
    }
    by_clip = defaultdict(list)
    for sample in dataset["samples"]:
        if sample["index"] in readable:
            by_clip[sample["source_clip"]].append(sample)

    videos = {Path(p).stem: Path(p) for p in dataset["clips"]}
    caches = {
        Path(p).name.lstrip(".").replace("_numbers_v1.npz", ""): Path(p)
        for p in dataset["caches"]
    }
    if args.embedding == "siglip":
        if args.team_model is None:
            parser.error("--team-model is required for the siglip embedding")
        model = TeamModel.load(args.team_model, device=args.device)
        encode = model.extract_features
    else:
        from handball_cv.embeddings.prtreid import PRTReIDBackend
        backend = PRTReIDBackend(
            source_root=args.prtreid_root, checkpoint=args.prtreid_checkpoint,
            device=args.device, feature_kind="global",
        )
        encode = backend.encode_images

    report = {
        "schema_version": 1, "embedding": args.embedding,
        "reid_cos_sim_min": REID_COS_SIM_MIN, "clips": {},
    }
    for clip, samples in sorted(by_clip.items()):
        video, cache = videos.get(clip), caches.get(clip)
        if video is None or cache is None:
            print(f"{clip}: no video/cache in the dataset manifest -- skipped")
            continue
        crops, kept = collect_crops(
            samples, video, cache, full_box=args.embedding == "prtreid"
        )
        if len(crops) < 4:
            continue
        embeddings = np.asarray(encode(crops), dtype=float)
        numbers = [readable[s["index"]] for s in kept]
        # Teams come from clustering these very embeddings: a split the embedding
        # itself cannot make is one re-ID could not have used either.
        from sklearn.cluster import KMeans
        teams = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
            embeddings
        ).tolist()
        stats = analyse_clip(cosine_matrix(embeddings), numbers, teams)
        report["clips"][clip] = stats
        same = stats["same_person_pairs"]
        other = stats["different_person_same_team_pairs"]
        rank1 = stats["rank1_same_team_gallery"]
        print(
            f"{clip[:34]:<34} n={stats['crops']:>3} "
            f"same {same['median']:.3f}  other(team) "
            f"{other['median'] if other else float('nan'):.3f} "
            f"(>{REID_COS_SIM_MIN}: {other['frac_above_reid_floor'] if other else float('nan'):.2f})  "
            f"rank1 {rank1['accuracy'] if rank1['accuracy'] is not None else float('nan'):.2f} "
            f"vs chance {rank1['chance'] if rank1['chance'] is not None else float('nan'):.2f}",
            flush=True,
        )

    totals = [c for c in report["clips"].values()]
    if totals:
        queries = sum(c["rank1_same_team_gallery"]["queries"] for c in totals)
        correct = sum(c["rank1_same_team_gallery"]["correct"] for c in totals)
        expected = sum(
            c["rank1_same_team_gallery"]["chance"]
            * c["rank1_same_team_gallery"]["queries"] for c in totals
        )
        report["overall_rank1_same_team"] = {
            "queries": queries, "correct": correct,
            "accuracy": correct / queries if queries else None,
            "chance": expected / queries if queries else None,
        }
        print(f"\noverall rank-1 within team: {correct}/{queries} = "
              f"{correct / queries:.2f}" if queries else "")
    if args.output:
        write_json_atomic(args.output, report)
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
