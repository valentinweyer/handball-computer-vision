"""Mine clean, number-vouched player fragments from existing tracking runs.

A *fragment* is one tracker id's uninterrupted presence: live frames of one
geometry id, cut at every identity break (`reid`, `suspected_id_switch`, the
same events `segment_verdicts` cuts at) and at every gap longer than
`MAX_GAP_FRAMES`. A fragment is labelled only by the OCR reads recorded on that
same physical track inside its own span -- never by a displayed, folded, linked
or backfilled family number. That is the source-track rule of the 2026-10-18
backfill fix (`tables._supported_track_segments`), made stricter here because a
training label needs more than a display label:

- the existing `NumberVoter` gates must pass on the fragment's own reads, and
- at least `MIN_OWN_AGREEING` exact reads, at least `MIN_AGREEMENT` share of
  the full-number reads, and no second number read twice.

Fragments are excluded, with the reason recorded, when the team is not settled
(goalkeeper rows, contested/provisional states, a team switch, a mixed team),
when the fragment touches a suspected id switch, when the final bundle withheld
or contradicted the number on its frames (arbitration loss or a contradiction),
or when another accepted fragment claims the same (team, number) while co-live.

High OCR agreement still does not prove identity: the accepted fragments are
proposals until the contact sheets have been reviewed (`review.json`).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from handball_cv.jersey.identity import IDENTITY_BREAK_EVENTS, NumberVoter

MAX_GAP_FRAMES = 25          # 1 s at 25 fps: longer absences open a new fragment
MIN_OWN_AGREEING = 5         # exact own reads of the verdict
MIN_AGREEMENT = 0.8          # share of the fragment's full-number reads
RIVAL_SHARE = 0.15           # a second number read >= 2x and >= this share of the verdict's reads
MIN_STABLE_SHARE = 0.9       # rows whose team_state is `stable`
MIN_DISPLAY_AGREEMENT = 0.9  # labelled rows whose final number equals the verdict
READ_CONTAINMENT = 0.9       # number box inside the player box, for legacy reads


@dataclass
class Fragment:
    video: str
    match: str
    track_id: int
    start: int
    end: int                       # inclusive
    frames: list = field(repr=False)
    team: int | None = None
    number: str | None = None      # own-read verdict
    reads: dict = field(default_factory=dict)   # value -> count, own reads only
    read_frames: list = field(default_factory=list, repr=False)
    read_provenance: str = "recorded"
    opened_by: str = "start"       # what began the fragment: start/gap/reid/suspected_id_switch
    closed_by: str = "end"
    status: str = "candidate"
    reasons: list = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.video}:t{self.track_id}:{self.start}-{self.end}"

    @property
    def identity(self) -> str | None:
        if self.number is None or self.team is None:
            return None
        return f"{self.match}|{self.video}|team{self.team}|#{self.number}"

    def to_json(self) -> dict:
        out = asdict(self)
        out.pop("frames")
        out.update(key=self.key, identity=self.identity, n_frames=len(self.frames))
        return out


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def _containment(inner, outer) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return ix * iy / area


def canonical_resolver(aliases: dict):
    def canonical(player_id: int) -> int:
        seen = set()
        while str(player_id) in aliases and player_id not in seen:
            seen.add(player_id)
            player_id = int(aliases[str(player_id)])
        return player_id
    return canonical


def attach_read_tracks(reads: list, tracks: pd.DataFrame, aliases: dict) -> tuple[list, str, Counter]:
    """Each read with the physical track that carried it, plus how that was known.

    Runs since the backfill fix record `track_id`. Older runs file a read under
    the identity at read time, which can already be a linked family id; there
    the track is reconstructed as the *single* box at that frame containing the
    number box, and is kept only if it resolves to the same final identity as
    the read. Anything ambiguous is dropped, never guessed.
    """
    if reads and all(row.get("track_id") is not None for row in reads):
        return [dict(row) for row in reads], "recorded", Counter(recorded=len(reads))
    canonical = canonical_resolver(aliases)
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    out, tally = [], Counter()
    for row in reads:
        if row.get("track_id") is not None:
            out.append(dict(row))
            tally["recorded"] += 1
            continue
        rows = by_frame.get(int(row["frame"]))
        if rows is None:
            tally["no_rows"] += 1
            continue
        hits = [int(r.track_id) for r in rows.itertuples()
                if _containment(row["box"], (r.x1, r.y1, r.x2, r.y2)) >= READ_CONTAINMENT]
        if len(hits) != 1:
            tally["ambiguous" if hits else "uncontained"] += 1
            continue
        if canonical(hits[0]) != canonical(int(row["player_id"])):
            tally["identity_mismatch"] += 1
            continue
        out.append({**row, "track_id": hits[0]})
        tally["reconstructed"] += 1
    return out, "reconstructed", tally


def split_fragments(video: str, match: str, tracks: pd.DataFrame, events: list,
                    max_gap: int = MAX_GAP_FRAMES) -> list[Fragment]:
    """Cut each tracker id's live frames at identity breaks and long gaps."""
    breaks: dict = {}
    for event in events:
        if event.get("type") in IDENTITY_BREAK_EVENTS and event.get("player_id") is not None:
            breaks.setdefault(int(event["player_id"]), {})[int(event["frame"])] = event["type"]
    fragments = []
    for track_id, group in tracks.groupby("track_id"):
        frames = np.unique(group["frame"].to_numpy())
        track_breaks = breaks.get(int(track_id), {})
        current, opened = [int(frames[0])], "start"
        for prev, frame in zip(frames[:-1], frames[1:]):
            prev, frame = int(prev), int(frame)
            cut = next((track_breaks[b] for b in sorted(track_breaks) if prev < b <= frame), None)
            if cut is None and frame - prev > max_gap:
                cut = "gap"
            if cut is not None:
                fragments.append(Fragment(video, match, int(track_id), current[0], current[-1],
                                          current, opened_by=opened, closed_by=cut))
                current, opened = [], cut
            current.append(frame)
        fragments.append(Fragment(video, match, int(track_id), current[0], current[-1],
                                  current, opened_by=opened, closed_by="end"))
    return fragments


def own_verdict(values: list) -> tuple[str | None, list]:
    """(verdict, reasons) from one fragment's own reads, in frame order."""
    voter = NumberVoter()
    for value in values:
        voter.observe(0, value)
    verdict = voter.best(0)[0]
    if verdict is None:
        return None, ["no_voter_verdict"]
    counts = Counter(v for v in values if v)
    full = {v: c for v, c in counts.items() if len(v) == len(verdict)}
    agreeing = counts.get(verdict, 0)
    reasons = []
    if agreeing < MIN_OWN_AGREEING:
        reasons.append(f"own_reads<{MIN_OWN_AGREEING}")
    if agreeing / max(1, sum(full.values())) < MIN_AGREEMENT:
        reasons.append("low_agreement")
    rivals = [v for v, c in counts.items() if v != verdict and c >= 2
              and c >= RIVAL_SHARE * agreeing and not (len(v) == 1 and verdict.endswith(v))]
    if rivals:
        reasons.append("rival_number:" + ",".join(sorted(rivals)))
    return verdict, reasons


def vouch_fragments(fragments: list, reads: list, tracks: pd.DataFrame, events: list) -> None:
    """Label and filter fragments in place; every rejection keeps its reasons."""
    reads_by_track: dict = {}
    for row in sorted(reads, key=lambda r: r["frame"]):
        reads_by_track.setdefault(int(row["track_id"]), []).append(row)
    team_switch = {(int(e["player_id"]), int(e["frame"])) for e in events
                   if e.get("type") == "team_switch" and e.get("player_id") is not None}
    rows_by_track = {t: g.set_index("frame") for t, g in tracks.groupby("track_id")}

    for fragment in fragments:
        own = [r for r in reads_by_track.get(fragment.track_id, ())
               if fragment.start <= int(r["frame"]) <= fragment.end]
        values = [r["value"] for r in own]
        fragment.reads = dict(Counter(values))
        fragment.read_frames = [int(r["frame"]) for r in own]
        verdict, reasons = own_verdict(values) if values else (None, ["no_own_reads"])
        fragment.number = verdict
        rows = rows_by_track[fragment.track_id].loc[fragment.start:fragment.end]
        if rows["is_goalkeeper"].any():
            reasons.append("goalkeeper")
        teams = rows["team"].value_counts()
        fragment.team = int(teams.index[0])
        if len(teams) > 1:
            reasons.append("mixed_team")
        if (rows["team_state"] == "stable").mean() < MIN_STABLE_SHARE:
            reasons.append("team_unsettled")
        if any(t == fragment.track_id and fragment.start <= f <= fragment.end for t, f in team_switch):
            reasons.append("team_switch")
        if "suspected_id_switch" in (fragment.opened_by, fragment.closed_by):
            reasons.append("suspected_id_switch")
        if verdict is not None:
            shown = rows["number"].dropna()
            if len(shown) < MIN_DISPLAY_AGREEMENT * len(rows):
                reasons.append("number_withheld")       # arbitration / duplicate guard
            elif (shown == verdict).mean() < MIN_DISPLAY_AGREEMENT:
                reasons.append("display_contradiction")
        fragment.reasons = reasons
        fragment.status = "accepted" if verdict is not None and not reasons else "excluded"

    # Two accepted fragments claiming one (team, number) while both on screen
    # cannot both be that player; neither is trusted.
    accepted = [f for f in fragments if f.status == "accepted"]
    for i, a in enumerate(accepted):
        for b in accepted[i + 1:]:
            if (a.team, a.number) == (b.team, b.number) and set(a.frames) & set(b.frames):
                for f in (a, b):
                    f.reasons.append("co_live_duplicate_number")
    for f in accepted:
        if f.reasons:
            f.status = "excluded"


@dataclass(frozen=True)
class VideoSpec:
    name: str
    match: str                 # underlying match: the split and leakage unit
    bundle: Path
    run: Path


def load_video(spec: VideoSpec) -> dict:
    """Tracks, number boxes, run summary and input hashes for one video."""
    run = json.loads(spec.run.read_text())
    manifest = json.loads((spec.bundle / "manifest.json").read_text())
    tracks = pd.read_parquet(spec.bundle / "tracks.parquet", columns=[
        "frame", "track_id", "player_id", "team", "team_state", "is_goalkeeper",
        "number", "x1", "y1", "x2", "y2"])
    detections = pd.read_parquet(spec.bundle / "detections.parquet")
    numbers = detections[detections["class_name"] == "number"]
    return {
        "spec": spec, "run": run, "manifest": manifest, "tracks": tracks, "numbers": numbers,
        "inputs": {
            "run": {"path": str(spec.run), "sha256": file_digest(spec.run)},
            "tracks": {"path": str(spec.bundle / "tracks.parquet"),
                       "sha256": file_digest(spec.bundle / "tracks.parquet")},
            "detections": {"path": str(spec.bundle / "detections.parquet"),
                           "sha256": file_digest(spec.bundle / "detections.parquet")},
            "video": {"path": str(spec.bundle / manifest["files"]["video"]),
                      "sha256": manifest["source"]["sha256"]},
        },
    }


def mine_video(loaded: dict) -> tuple[list, dict]:
    spec, run, tracks = loaded["spec"], loaded["run"], loaded["tracks"]
    reads, provenance, tally = attach_read_tracks(run["number_reads"], tracks,
                                                  run.get("player_aliases", {}))
    fragments = split_fragments(spec.name, spec.match, tracks, run["identity_events"])
    for f in fragments:
        f.read_provenance = provenance
    vouch_fragments(fragments, reads, tracks, run["identity_events"])
    summary = {
        "video": spec.name, "match": spec.match, "read_provenance": provenance,
        "read_attachment": dict(tally), "fragments": len(fragments),
        "accepted": sum(f.status == "accepted" for f in fragments),
        "exclusion_reasons": dict(Counter(r.split(":")[0] for f in fragments
                                          if f.number is not None for r in f.reasons)),
        "identities": len({f.identity for f in fragments if f.status == "accepted"}),
    }
    return fragments, summary


# --- crop sampling -----------------------------------------------------------

MIN_BOX_HEIGHT = 64
MAX_OTHER_COVER = 0.10       # share of this box covered by any other tracked box
BORDER_MARGIN = 4
NUMBER_IN_BOX = 0.6          # number box containment to count as this player's digits


def sample_crops(fragment: Fragment, tracks_by_frame: dict, numbers_by_frame: dict,
                 detection_frames: set, width: int, height: int,
                 every: int = 10, cap: int = 40, referees_by_frame: dict | None = None) -> tuple[list, Counter]:
    """Crop records for one fragment, at detector frames only.

    Restricting to detector frames means "no number box" really means the
    number detector found none there, which is what digit masking relies on.
    Crops whose box is overlapped by another tracked person or a detected
    referee (referees are not tracked) are dropped, so a neighbour's
    appearance (or ownership doubt) never becomes a label.
    """
    candidates = [f for f in fragment.frames if f in detection_frames]
    picked, last = [], -10**9
    for f in candidates:
        if f - last >= every:
            picked.append(f)
            last = f
    if len(picked) > cap:
        picked = [picked[i] for i in np.linspace(0, len(picked) - 1, cap).round().astype(int)]
    records, dropped = [], Counter()
    for f in picked:
        rows = tracks_by_frame[f]
        me = rows[rows["track_id"] == fragment.track_id].iloc[0]
        box = np.array([me.x1, me.y1, me.x2, me.y2], dtype=float)
        if box[3] - box[1] < MIN_BOX_HEIGHT:
            dropped["small"] += 1
            continue
        if box[0] < BORDER_MARGIN or box[1] < BORDER_MARGIN or box[2] > width - BORDER_MARGIN \
                or box[3] > height - BORDER_MARGIN:
            dropped["border"] += 1
            continue
        others = [(o.x1, o.y1, o.x2, o.y2) for o in rows[rows["track_id"] != fragment.track_id].itertuples()]
        others += [(o.x1, o.y1, o.x2, o.y2) for o in (referees_by_frame or {}).get(f, ())]
        cover = max((_containment(box, o) for o in others), default=0.0)
        if cover > MAX_OTHER_COVER:
            dropped["overlap"] += 1
            continue
        digits = [list(map(float, (n.x1, n.y1, n.x2, n.y2)))
                  for n in numbers_by_frame.get(f, ()) if _containment((n.x1, n.y1, n.x2, n.y2), box)
                  >= NUMBER_IN_BOX]
        records.append({
            "video": fragment.video, "match": fragment.match, "fragment": fragment.key,
            "identity": fragment.identity, "team": fragment.team, "number": fragment.number,
            "track_id": fragment.track_id, "frame": int(f), "box": box.tolist(),
            "digit_boxes": digits, "own_read_here": f in set(fragment.read_frames),
        })
    return records, dropped
