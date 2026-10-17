"""Assemble the bundle's tables from what the extraction stages produce.

Inputs are the stage outputs as they exist today: the tracking pass's run
summary (the dict `render_full_pipeline.render_sam2` writes as JSON), its
geometry cache, per-frame `CourtFit`s, the court keypoints, and the detection
cache. Nothing here runs a model; this is joining and bookkeeping, which is why
a bundle can be rebuilt in seconds when only the assembly changes.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np
import pandas as pd

from handball_cv.court.keypoints import KEYPOINT_TO_VERTEX
from handball_cv.jersey.identity import segment_verdicts
from handball_cv.visualization.palette import GOALKEEPER, TEAM_STATE

__all__ = [
    "canonical_resolver", "court_table", "detections_table", "events_table",
    "keypoints_table", "players_table", "reads_table", "tracks_table",
]

# Project-wide detection classes (see scripts.render_raw_team_classification).
CLASS_NAMES = {1: "goalkeeper", 2: "player", 3: "referee", 4: "number"}


def canonical_resolver(aliases: dict):
    """`player_id -> the identity it was folded into`, following alias chains."""
    table = {int(k): int(v) for k, v in aliases.items()}

    def canonical(player_id: int) -> int:
        seen = set()
        while player_id in table and player_id not in seen:
            seen.add(player_id)
            player_id = table[player_id]
        return player_id

    return canonical


def _segments_by_player(run: dict, ocr_every: int, fold_links: bool = False) -> dict:
    """Per-identity number segments, keyed by player id.

    Unfolded, these are exactly the segments `--redraw` uses: live frames keyed
    by the tracker's own ids, reads by the identity they were filed under.

    `fold_links` first maps every id through the run's number links, so a
    fragment linked into an identity contributes its frames, reads and re-ID
    breaks to that identity. A link asserts the two are one person who was never
    on screen twice at once -- same team, same resolved number, never co-live --
    so the settled number describes the fragment's frames too. Unfolded, those
    frames belong to an id with no reads of its own after the link and are never
    labelled: on a 10-minute Kiel-Lemgo window, 81% of the frames of identities
    that resolved a number were such fragments, and none of them carried it.
    """
    canonical = canonical_resolver(run.get("player_aliases", {}) if fold_links else {})
    live_frames: dict = {}
    for frame, ids in run["frame_players"].items():
        for player_id in ids:
            live_frames.setdefault(canonical(int(player_id)), []).append(int(frame))
    reads = [{**row, "player_id": canonical(int(row["player_id"]))} for row in run["number_reads"]]
    # Every id an event names goes through the same fold, so a link folded here
    # already is a no-op when the label replay reaches it.
    events = [{**event, **{key: canonical(int(event[key])) for key in ("player_id", "into_player_id")
                           if event.get(key) is not None}}
              for event in run["identity_events"]]
    teams = {}
    for player_id, team in run["player_teams"].items():
        root = canonical(int(player_id))
        teams[root] = run["player_teams"].get(str(root), team)  # a fold takes its root's team
    segments = segment_verdicts(reads, events, live_frames, teams=teams, ocr_every=ocr_every)
    by_player: dict = {}
    for segment in segments:
        by_player.setdefault(segment.player_id, []).append(segment)
    return by_player


def one_place_per_identity(track_ids: list[int], canonical) -> list[int]:
    """Each row's identity, never the same identity on two boxes in one frame.

    A number link is checked when it is made -- the two ids had never been on
    screen together -- but nothing undoes it if they are later. Folding through
    the final aliases would then put one identity in two places at once (690
    frames on a 10-minute Kiel-Lemgo window). Where that happens, the box that
    is the identity's own tracker id keeps it and the others fall back to their
    own ids, which carry no folded number.
    """
    roots = [canonical(t) for t in track_ids]
    counts = Counter(roots)
    return [t if counts[root] > 1 and t != root else root for t, root in zip(track_ids, roots)]


def one_number_per_team(numbers: list, teams, goalkeepers) -> list:
    """Withhold a number shown on two field players of one team in the same frame.

    One of them is wrong and nothing here says which, so neither is shown --
    abstaining beats a confident wrong label. The run's own arbitration does
    this live; a backfilled verdict reaches back over frames it never saw.
    """
    seen = Counter((int(team), number) for number, team, keeper in zip(numbers, teams, goalkeepers)
                   if number is not None and not keeper)
    return [None if number is not None and not keeper and seen[(int(team), number)] > 1 else number
            for number, team, keeper in zip(numbers, teams, goalkeepers)]


def tracks_table(run: dict, geometry, fits: list, fps: float, ocr_every: int,
                 team_confidence: np.ndarray | None = None) -> pd.DataFrame:
    """One row per tracked player per frame.

    `number` is the backfilled verdict -- the segment's settled number, reaching
    back over the frames before it was decided -- with number links folded in
    (see `_segments_by_player`): a fragment linked into an identity carries that
    identity's number. Without links this is exactly what `--redraw` draws.
    `number_live` is what the render drew at that frame: the run's own label
    replay, each number link applied from the frame it was made, as
    `--redraw --redraw-causal` draws it. Two invariants hold
    per frame: an identity is on at most one box (`one_place_per_identity`),
    and a number is on at most one field player per team
    (`one_number_per_team`); where either would break, the number is withheld. `court_x/court_y`
    are the box's bottom-centre through that frame's court fit, NaN where the
    fit abstained. `team_confidence`, when the tracking pass recorded it, is
    one value per geometry row; without it the column is NaN.
    """
    canonical = canonical_resolver(run.get("player_aliases", {}))
    folded = _segments_by_player(run, ocr_every, fold_links=True)
    unfolded = _segments_by_player(run, ocr_every)

    def segment_at(by_player: dict, player_id: int, frame: int):
        return next((s for s in by_player.get(player_id, ())
                     if s.start <= frame <= s.end), None)

    columns: dict[str, list] = {name: [] for name in (
        "frame", "time_s", "player_id", "track_id", "team", "team_state", "team_confidence",
        "is_goalkeeper", "number", "number_live", "x1", "y1", "x2", "y2", "foot_x", "foot_y",
        "court_x", "court_y", "court_fit_source", "mask_row")}
    for frame in geometry.frame_indices:
        lo = geometry.row_offset(frame)
        ids, boxes, colors, teams = geometry.rows(frame)
        if not len(ids):
            continue
        fit = fits[frame] if frame < len(fits) else None
        feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
        if fit is not None and fit.usable:
            court = fit.to_court(feet)
        else:
            court = np.full((len(ids), 2), np.nan, dtype=np.float32)
        player_ids = one_place_per_identity([int(t) for t in ids], canonical)
        keepers = [int(c) == GOALKEEPER for c in colors]
        numbers = []
        for player_id in player_ids:
            segment = segment_at(folded, player_id, frame)
            numbers.append(segment.value_at(frame) if segment else None)
        numbers = one_number_per_team(numbers, teams, keepers)
        for row, track_id in enumerate(ids):
            player_id = player_ids[row]
            # As the render labelled it: by the tracker's own id, read through links.
            live_segment = segment_at(unfolded, int(track_id), frame)
            color = int(colors[row])
            columns["frame"].append(frame)
            columns["time_s"].append(frame / fps)
            columns["player_id"].append(player_id)
            columns["track_id"].append(int(track_id))
            columns["team"].append(int(teams[row]))
            columns["team_state"].append(TEAM_STATE.get(color))
            columns["team_confidence"].append(
                np.nan if team_confidence is None else float(team_confidence[lo + row]))
            columns["is_goalkeeper"].append(color == GOALKEEPER)
            columns["number"].append(numbers[row])
            columns["number_live"].append(live_segment.live_at(frame) if live_segment else None)
            for name, value in zip(("x1", "y1", "x2", "y2"), boxes[row]):
                columns[name].append(float(value))
            columns["foot_x"].append(float(feet[row, 0]))
            columns["foot_y"].append(float(feet[row, 1]))
            columns["court_x"].append(float(court[row, 0]))
            columns["court_y"].append(float(court[row, 1]))
            columns["court_fit_source"].append(fit.source if fit is not None else "none")
            columns["mask_row"].append(lo + row)
    frame = pd.DataFrame(columns)
    return frame.astype({
        "frame": "int32", "player_id": "int32", "track_id": "int32", "team": "int8",
        "x1": "float32", "y1": "float32", "x2": "float32", "y2": "float32",
        "foot_x": "float32", "foot_y": "float32", "court_x": "float32", "court_y": "float32",
        "mask_row": "int64",
    })


def players_table(run: dict, tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per identity after linking. `number` is the run's final verdict."""
    canonical = canonical_resolver(run.get("player_aliases", {}))
    folded: dict = {}
    for alias in run.get("player_aliases", {}):
        folded.setdefault(canonical(int(alias)), []).append(int(alias))
    reids: dict = {}
    for event in run["identity_events"]:
        if event["type"] == "reid":
            reids[canonical(int(event["player_id"]))] = reids.get(canonical(int(event["player_id"])), 0) + 1
    reads: dict = {}
    for read in run["number_reads"]:
        reads[int(read["player_id"])] = reads.get(int(read["player_id"]), 0) + 1

    rows = []
    for player_id, group in tracks.groupby("player_id", sort=True):
        player_id = int(player_id)
        rows.append({
            "player_id": player_id,
            "team": int(run["player_teams"].get(str(player_id), -1)),
            "is_goalkeeper": bool(group["is_goalkeeper"].mean() > 0.5),
            "number": run["numbers_resolved"].get(str(player_id)),
            "first_frame": int(group["frame"].min()),
            "last_frame": int(group["frame"].max()),
            "frames_visible": int(len(group)),
            "reid_count": reids.get(player_id, 0),
            "folded_ids": ",".join(str(i) for i in sorted(folded.get(player_id, []))),
            "reads": reads.get(player_id, 0),
            "votes": json.dumps(run["number_votes_raw"].get(str(player_id), {})),
        })
    return pd.DataFrame(rows).astype({"player_id": "int32", "team": "int8"})


def reads_table(run: dict) -> pd.DataFrame:
    rows = [{
        "frame": read["frame"], "player_id": read["player_id"], "value": read["value"],
        # Optional additive provenance: older runs cannot recover this from a
        # canonical player_id, so leave it missing rather than guessing.
        "track_id": read.get("track_id"),
        "confidence": read.get("confidence", np.nan), "number_row": read["number_row"],
        "x1": read["box"][0], "y1": read["box"][1], "x2": read["box"][2], "y2": read["box"][3],
    } for read in run["number_reads"]]
    columns = ["frame", "player_id", "track_id", "value", "confidence", "number_row", "x1", "y1", "x2", "y2"]
    return pd.DataFrame(rows, columns=columns).astype({
        "frame": "int32", "player_id": "int32", "track_id": "Int32",
    })


def events_table(run: dict) -> pd.DataFrame:
    base = ["frame", "type", "player_id", "fragment_id", "team_id"]
    frame = pd.DataFrame(run["identity_events"])
    for column in base:
        if column not in frame:
            frame[column] = pd.NA
    return frame[base + [c for c in frame.columns if c not in base]].astype(
        {"frame": "int32", "player_id": "Int32", "fragment_id": "Int32", "team_id": "Int8"})


def court_table(fits: list) -> pd.DataFrame:
    """One row per frame. `h*` maps court cm -> image px, `inv*` image -> court."""
    rows = []
    for frame, fit in enumerate(fits):
        row = {"frame": frame, "source": fit.source, "court_in_view": fit.court_in_view,
               "age": fit.age, "inliers": fit.inliers,
               "reprojection_px": fit.reprojection_px,
               "coverage_x_cm": fit.coverage_cm[0], "coverage_y_cm": fit.coverage_cm[1]}
        for prefix, matrix in (("h", fit.homography), ("inv", fit.inverse)):
            values = np.full(9, np.nan) if matrix is None else np.asarray(matrix, float).ravel()
            row.update({f"{prefix}{i}": values[i] for i in range(9)})
        rows.append(row)
    return pd.DataFrame(rows).astype({"frame": "int32", "age": "int32", "inliers": "int32"})


def keypoints_table(keypoints: list) -> pd.DataFrame:
    rows = [(frame, int(slot), KEYPOINT_TO_VERTEX[int(slot)], x, y, conf)
            for frame, points in enumerate(keypoints) for slot, x, y, conf in points]
    return pd.DataFrame(rows, columns=["frame", "slot", "vertex", "x", "y", "confidence"]).astype(
        {"frame": "int32", "slot": "int8", "vertex": "int8",
         "x": "float32", "y": "float32", "confidence": "float32"})


def detections_table(cache: dict) -> pd.DataFrame:
    offsets = np.asarray(cache["offsets"])
    frames = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    boxes = np.asarray(cache["boxes"], dtype=np.float32).reshape(-1, 4)
    class_id = np.asarray(cache["class_id"]).astype(np.int8)
    return pd.DataFrame({
        "frame": frames.astype(np.int32),
        "class_id": class_id,
        "class_name": [CLASS_NAMES.get(int(c), str(int(c))) for c in class_id],
        "confidence": np.asarray(cache["confidence"], dtype=np.float32),
        "x1": boxes[:, 0], "y1": boxes[:, 1], "x2": boxes[:, 2], "y2": boxes[:, 3],
    })
