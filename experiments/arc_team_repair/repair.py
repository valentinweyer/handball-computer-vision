"""Optional offline team-confirmed repair of overlap identity errors.

Input team reads are independent, qualified frame-local observations. This
module never predicts teams, changes boxes, feeds repaired identities back
into observations, or connects new track births. Default behaviour is exact
pass-through. Uncertain encounters are logged and left alone.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class RepairConfig:
    overlap_iou: float = 0.10
    contact_gap: int = 2
    min_contact_frames: int = 2
    before_window: int = 40
    after_window: int = 45
    min_votes: int = 5
    max_votes: int = 10
    vote_purity: float = 0.90
    suffix_purity: float = 0.95
    min_switch_gain: float = 0.05
    min_boundary_margin: float = 0.01
    handover_bridge_frames: int = 3
    handover_min_iou: float = 0.30


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9)


def parse_lines(lines):
    by_frame = defaultdict(dict)
    for line in lines:
        p = line.split(",")
        f, tid = int(p[0]) - 1, int(p[1])
        x, y, w, h = map(float, p[2:6])
        if tid in by_frame[f]:
            raise ValueError(f"duplicate track {tid} in frame {f}")
        by_frame[f][tid] = np.array([x, y, x + w, y + h])
    return by_frame


def encounters(by_frame, cfg):
    touches = defaultdict(list)
    for f, boxes in sorted(by_frame.items()):
        for a, b in combinations(sorted(boxes), 2):
            if iou(boxes[a], boxes[b]) >= cfg.overlap_iou:
                touches[a, b].append(f)
    result = []
    for (a, b), times in touches.items():
        groups = []
        for f in times:
            if groups and f - groups[-1][-1] <= cfg.contact_gap:
                groups[-1].append(f)
            else:
                groups.append([f])
        for group in groups:
            if len(group) >= cfg.min_contact_frames:
                result.append({"pair": [a, b], "start": group[0], "end": group[-1],
                               "contact_frames": len(group)})
    return sorted(result, key=lambda e: (e["end"], e["start"], e["pair"]))


def _anchor(reads, tid, low, high, cfg, before):
    votes = sorted((f, team) for (f, track), team in reads.items()
                   if track == tid and low <= f <= high and team in (0, 1))
    votes = votes[-cfg.max_votes:] if before else votes[:cfg.max_votes]
    if len(votes) < cfg.min_votes:
        return None
    team, count = Counter(t for _, t in votes).most_common(1)[0]
    if count / len(votes) < cfg.vote_purity:
        return None
    return {"team": team, "votes": len(votes), "purity": count / len(votes),
            "first": votes[0][0], "last": votes[-1][0]}


def _switch_boundary(by_frame, a, b, start, end, cfg):
    choices = []
    for f in range(max(1, start), end + 2):
        previous, current = by_frame.get(f - 1, {}), by_frame.get(f, {})
        if not all(t in previous and t in current for t in (a, b)):
            continue
        stay = iou(previous[a], current[a]) + iou(previous[b], current[b])
        cross = iou(previous[a], current[b]) + iou(previous[b], current[a])
        choices.append((cross - stay, f))
    choices.sort(reverse=True)
    if not choices or choices[0][0] < cfg.min_switch_gain:
        return None
    if len(choices) > 1 and choices[0][0] - choices[1][0] < cfg.min_boundary_margin:
        return None
    return {"frame": choices[0][1], "gain": float(choices[0][0]),
            "margin": float(choices[0][0] - choices[1][0]) if len(choices) > 1 else None}


def _handovers(by_frame, reads, events, cfg, used):
    """Reconnect a wrong-team tail only to a uniquely supported lost partner.

    Unlike a live pair swap, this does not guess routes inside a crowded
    overlap. The boundary is the first qualified clear team observation
    after the partner's final box, once later observations confirm it.
    A short geometric bridge and clear independent team anchors are required.
    All plausible candidates are collected before choosing any: competing
    partners/episodes abstain, rather than winning by iteration order.
    """
    last = {}
    for f, boxes in sorted(by_frame.items()):
        for tid in boxes:
            last[tid] = f
    audits, candidates, plausible = [], [], []
    for event in events:
        a, b = event["pair"]
        if last[a] == last[b]:
            continue
        lost, survivor = (a, b) if last[a] < last[b] else (b, a)
        # The lost track must end in this contact, never return later, and
        # the survivor must continue immediately. No new births are linked.
        if last[lost] != event["end"] or survivor not in by_frame.get(last[lost] + 1, {}):
            continue
        e = dict(event, kind="lost_partner", lost=lost, survivor=survivor)
        audits.append(e)
        if used.intersection((lost, survivor)):
            e["decision"] = "track_already_repaired"
            continue
        pre = [_anchor(reads, t, event["start"] - cfg.before_window,
                       event["start"] - 1, cfg, True) for t in (lost, survivor)]
        post = _anchor(reads, survivor, event["end"] + 1,
                       event["end"] + cfg.after_window, cfg, False)
        e["pre"], e["post"] = pre, post
        end = event["end"]
        bridge = [(iou(by_frame[end][lost], by_frame[f][survivor]), f)
                  for f in range(end + 1, end + cfg.handover_bridge_frames + 1)
                  if survivor in by_frame.get(f, {})]
        score, frame = max(bridge, default=(0., end + 1))
        e["bridge"] = {"iou": float(score), "frame": frame}
        if score < cfg.handover_min_iou:
            e["decision"] = "unsupported_geometric_bridge"
            continue
        # Missing colour evidence cannot rule out a spatially plausible
        # competing route. A clearly different destination team can.
        if pre[0] is None or post is None or pre[0]["team"] == post["team"]:
            plausible.append(e)
        if any(x is None for x in pre) or post is None:
            e["decision"] = "insufficient_clear_team_evidence"
            continue
        if pre[0]["team"] == pre[1]["team"] or post["team"] != pre[0]["team"]:
            e["decision"] = "no_supported_team_handover"
            continue
        future = [team for (f, t), team in reads.items()
                  if t == survivor and f >= post["first"] and team in (0, 1)]
        if sum(t == post["team"] for t in future) / len(future) < cfg.suffix_purity:
            e["decision"] = "later_team_contradiction"
            continue
        first_support = min(f for (f, t), team in reads.items()
                            if t == survivor and post["first"] <= f <= post["last"]
                            and team == post["team"])
        e["boundary"] = {"frame": first_support}
        candidates.append(e)
    accepted = []
    for e in candidates:
        tids = {e["lost"], e["survivor"]}
        if any(other is not e and tids.intersection((other["lost"], other["survivor"]))
               for other in plausible):
            e["decision"] = "ambiguous_handover_partner"
            continue
        e["decision"] = "repair"
        accepted.append(e)
    return accepted, audits


def repair_tracks(lines, reads, *, enabled=False, config=None):
    """Return MOT lines and an audit. Only identity fields may change.

    A correction swaps two live IDs at a supported crossing, or reconnects
    a wrong-team tail to its uniquely supported lost partner after contact.
    One accepted correction per involved track prevents untested
    composition across several contacts. Later contrary clear team evidence
    blocks a correction rather than extending it blindly through the video.
    """
    cfg = config or RepairConfig()
    if not enabled:
        return list(lines), {"enabled": False, "corrections": [], "encounters": []}
    by_frame = parse_lines(lines)
    audit = {"enabled": True, "config": asdict(cfg), "corrections": [], "encounters": []}
    used = set()
    proposals = []
    events = encounters(by_frame, cfg)
    for raw_event in events:
        e = dict(raw_event)
        audit["encounters"].append(e)
        a, b = e["pair"]
        if used.intersection((a, b)):
            e["decision"] = "track_already_repaired"
            continue
        # Any third participant makes this a multi-route problem, outside
        # the narrow two-person prototype.
        crowded = False
        missing = False
        for f in range(e["start"], e["end"] + 1):
            boxes = by_frame.get(f, {})
            if a not in boxes or b not in boxes:
                missing = True
                break
            for other in set(boxes) - {a, b}:
                if max(iou(boxes[a], boxes[other]), iou(boxes[b], boxes[other])) >= cfg.overlap_iou:
                    crowded = True
                    break
        if missing or crowded:
            e["decision"] = "missing_track" if missing else "third_participant"
            continue
        pre = [_anchor(reads, t, e["start"] - cfg.before_window, e["start"] - 1,
                       cfg, True) for t in (a, b)]
        post = [_anchor(reads, t, e["end"] + 1, e["end"] + cfg.after_window,
                        cfg, False) for t in (a, b)]
        e["pre"], e["post"] = pre, post
        if any(x is None for x in pre + post):
            e["decision"] = "insufficient_clear_team_evidence"
            continue
        if pre[0]["team"] == pre[1]["team"]:
            e["decision"] = "same_team"
            continue
        if [x["team"] for x in pre] == [x["team"] for x in post]:
            e["decision"] = "already_consistent"
            continue
        if [x["team"] for x in post] != [pre[1]["team"], pre[0]["team"]]:
            e["decision"] = "not_a_mutual_team_swap"
            continue
        stable_suffix = True
        for tid, anchor in zip((a, b), post):
            future = [team for (f, t), team in reads.items()
                      if t == tid and f >= anchor["first"] and team in (0, 1)]
            if sum(t == anchor["team"] for t in future) / len(future) < cfg.suffix_purity:
                stable_suffix = False
        if not stable_suffix:
            e["decision"] = "later_team_contradiction"
            continue
        boundary = _switch_boundary(by_frame, a, b, e["start"], e["end"], cfg)
        if boundary is None:
            e["decision"] = "uncertain_switch_boundary"
            continue
        e["decision"] = "repair"
        e["boundary"] = boundary
        proposals.append((a, b, boundary["frame"]))
        used.update((a, b))
        audit["corrections"].append(dict(e))
    handovers, audit["handovers"] = _handovers(by_frame, reads, events, cfg, used)
    audit["corrections"].extend(dict(e) for e in handovers)
    output = []
    changed = 0
    for line in lines:
        parts = line.split(",")
        f, tid = int(parts[0]) - 1, int(parts[1])
        target = tid
        for a, b, boundary in proposals:
            if f >= boundary:
                if tid == a:
                    target = b
                elif tid == b:
                    target = a
        for e in handovers:
            if f >= e["boundary"]["frame"] and tid == e["survivor"]:
                target = e["lost"]
        if target != tid:
            parts[1] = str(target)
            line = ",".join(parts)
            changed += 1
        output.append(line)
    audit["changed_rows"] = changed
    assert len(output) == len(lines)
    parse_lines(output)   # No duplicate IDs may be created in any frame.
    return output, audit
