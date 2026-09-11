"""Jersey-number identity signal.

Number boxes are class 4 of the player-detection model already being called for
players -- no separate detector, no extra inference call to find them. OCR is a
EasyOCR recognizes tight native-size grayscale crops locally. The original
basketball-tuned SmolVLM2 path systematically confused small handball ``13``
crops with ``8``/``18``, so reads remain confidence-gated and voted over time.

Numbers are matched to player masks by mask IoS (intersection over the *smaller*
area), not IoU: a number crop is small and, when correctly matched, fully
contained within its player's mask, so IoS saturates to 1.0 while IoU would stay
low regardless of match quality. Ported from the notebook this pipeline never
wired in (handball_ai_how_to_..._players.ipynb, cells 83-95).

Two things are deliberately NOT ported from that notebook:

  - `sports.common.temporal.ConsecutiveValueTracker` locks a value permanently
    with no un-lock, which is wrong once re-ID or a hard reset can legitimately
    move a physical player to a new tracker-level id.
  - its `player_idx = [i + 1 for i in player_idx]` assumes
    `index + 1 == tracker_id`, true only because SAM2 was seeded once with
    contiguous ids from `np.arange(1, N+1)`. That assumption is false once
    TrackManager (or McByte) allocates and retires ids out of order.

`NumberVoter` here keys on the caller's own stable identity (obj_id / player_id)
and can revise its answer as evidence accumulates.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv

NUMBER_CLASS_ID   = 4
NUMBER_PROMPT     = "Read the number."  # retained for notebook/API compatibility
NUMBER_CROP_PAD   = 0
EASY_OCR_MIN_CONFIDENCE = 0.5
MATCH_IOS_MIN     = 0.9
OCR_EVERY_N_FRAMES = 5          # number detection is expensive; sample, don't run every frame
# IHF numbering: 1-99. "0"/"00" are not legal, and a leading zero ("07") is not
# how a jersey is printed, so any of those coming back from the reader is a
# misread or a false number detection rather than a player.
MIN_JERSEY_NUMBER, MAX_JERSEY_NUMBER = 1, 99


def is_valid_number(value: str) -> bool:
    """Whether a read could be a real handball jersey number."""
    if not value.isdigit() or len(value) > 2:
        return False
    if len(value) == 2 and value[0] == "0":
        return False
    return MIN_JERSEY_NUMBER <= int(value) <= MAX_JERSEY_NUMBER


def extract_number_boxes(detections: sv.Detections) -> sv.Detections:
    """Split class-4 (jersey number) detections out of a player-model result."""
    return detections[detections.class_id == NUMBER_CLASS_ID]


def match_numbers_to_players(
    player_masks: np.ndarray,
    number_xyxy: np.ndarray,
    frame_shape: tuple,
) -> list:
    """[(player_row, number_row), ...] pairs above MATCH_IOS_MIN, best match first.

    frame_shape: (height, width) -- note sv.xyxy_to_mask wants resolution_wh as
    (width, height), the reverse order.
    """
    if len(player_masks) == 0 or len(number_xyxy) == 0:
        return []
    h, w = frame_shape[:2]
    number_masks = sv.xyxy_to_mask(boxes=np.asarray(number_xyxy).reshape(-1, 4), resolution_wh=(w, h))
    iou = sv.mask_iou_batch(
        masks_true=player_masks, masks_detection=number_masks, overlap_metric=sv.OverlapMetric.IOS
    )
    rows, cols = np.where(iou > MATCH_IOS_MIN)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    pairs.sort(key=lambda rc: iou[rc[0], rc[1]], reverse=True)
    return pairs


def read_numbers(model, frame_rgb: np.ndarray, number_xyxy: np.ndarray) -> list:
    """Recognize tight, native-size number crops with EasyOCR.

    The number detector already supplies a tight box. Padding and square
    stretching made the 25-40 px handball digits less legible, while the
    basketball-tuned VLM systematically read a labeled ``13`` as ``8``/``18``.
    Returns ``''`` for a low-confidence or failed read.
    """
    number_xyxy = np.asarray(number_xyxy).reshape(-1, 4)
    if len(number_xyxy) == 0:
        return []
    h, w = frame_rgb.shape[:2]
    padded = sv.clip_boxes(
        sv.pad_boxes(xyxy=number_xyxy, px=NUMBER_CROP_PAD, py=NUMBER_CROP_PAD), (w, h)
    )
    crops_grey = [
        cv2.cvtColor(sv.crop_image(frame_rgb, box), cv2.COLOR_RGB2GRAY)
        for box in padded
    ]
    out = []
    for crop in crops_grey:
        try:
            crop_h, crop_w = crop.shape
            result = model.recognize(
                crop,
                horizontal_list=[[0, crop_w, 0, crop_h]],
                free_list=[],
                allowlist="0123456789",
                detail=1,
                decoder="beamsearch",
                beamWidth=10,
                reformat=False,
            )
            value, confidence = result[0][1], float(result[0][2])
            out.append(value if confidence >= EASY_OCR_MIN_CONFIDENCE else "")
        except Exception:
            out.append("")
    return out


DOCTR_MIN_CONFIDENCE = 0.5


def read_numbers_doctr(predictor, frame_rgb: np.ndarray, number_xyxy: np.ndarray) -> list:
    """Recognize the same tight crops with a docTR scene-text recogniser.

    Deliberately not routed through `read_numbers`: that path converts to
    greyscale for EasyOCR, and these models are trained on colour text lines.
    They also want the *tight* crop specifically -- measured on the 1080p
    evaluation set, `parseq` scores 0.62 accuracy on the tight crop and 0.04 on
    the padded context crop, the reverse of the VLM's preference.

    All boxes in a frame go through the predictor in one batch; per-crop calls
    dominate runtime otherwise. The recogniser emits free text, so a read only
    counts when it is a legal jersey number, matching what the voter accepts.
    """
    number_xyxy = np.asarray(number_xyxy).reshape(-1, 4)
    if len(number_xyxy) == 0:
        return []
    height, width = frame_rgb.shape[:2]
    boxes = sv.clip_boxes(
        sv.pad_boxes(xyxy=number_xyxy, px=NUMBER_CROP_PAD, py=NUMBER_CROP_PAD),
        (width, height),
    )
    crops, kept = [], []
    for position, box in enumerate(boxes):
        crop = sv.crop_image(frame_rgb, box)
        if crop.size and crop.shape[0] >= 2 and crop.shape[1] >= 2:
            crops.append(crop)
            kept.append(position)

    out = [""] * len(boxes)
    if not crops:
        return out
    # Deliberately not wrapped in try/except. An empty read means "this crop is
    # not a legible number", and a broken predictor must never be able to say
    # that: a misconfigured caller once passed predictor=None here and the
    # resulting crash was swallowed into 0 reads over 99 frames, which looked
    # like a measurement rather than a bug. Failures belong at the call site.
    results = predictor(crops)
    for position, result in zip(kept, results):
        text, confidence = str(result[0]).strip(), float(result[1])
        if confidence >= DOCTR_MIN_CONFIDENCE and is_valid_number(text):
            out[position] = text
    return out


PARSEQ_MIN_CONFIDENCE = 0.5


def read_numbers_parseq(reader, frame_rgb: np.ndarray, number_xyxy: np.ndarray) -> list:
    """Recognize the tight crops with the original `baudm/parseq` weights.

    Same architecture as `read_numbers_doctr`, different training. Measured on
    the 323 human-readable crops of the 1080p evaluation set, scored identically:

        reader                    accuracy   selective   abstention
        docTR parseq                 0.622       0.817         0.88
        baudm parseq (this)          0.858       0.958         0.89

    Paired McNemar b=4 c=80, p=2e-19 -- 80 crops docTR misses that this reads,
    against 4 the other way, at the same ~1.6 ms/crop. The failures it removes
    are the ones that were driving pipeline errors: 22 read as 2, 17 and 10 and
    11 abstained on, 7 read as 1. docTR trains its own recognisers largely on
    document text; these weights are trained on scene-text benchmarks, which
    is far closer to a number on a moving shirt.

    Fine-tuning on other sports was measured and does not help: the same repo's
    hockey weights score 0.851 with much worse abstention (0.49), and its
    SoccerNet weights collapse to 0.266.

    `reader` is the `(model, transform)` pair from `_jersey_parseq.load_jersey_parseq`.
    Not wrapped in try/except, for the reason `read_numbers_doctr` gives.
    """
    number_xyxy = np.asarray(number_xyxy).reshape(-1, 4)
    if len(number_xyxy) == 0:
        return []
    height, width = frame_rgb.shape[:2]
    boxes = sv.clip_boxes(
        sv.pad_boxes(xyxy=number_xyxy, px=NUMBER_CROP_PAD, py=NUMBER_CROP_PAD),
        (width, height),
    )
    crops, kept = [], []
    for position, box in enumerate(boxes):
        crop = sv.crop_image(frame_rgb, box)
        if crop.size and crop.shape[0] >= 2 and crop.shape[1] >= 2:
            crops.append(crop)
            kept.append(position)

    out = [""] * len(boxes)
    if not crops:
        return out
    model, transform = reader
    from handball_cv.jersey.parseq_backend import read_crops

    for position, (text, confidence) in zip(kept, read_crops(model, transform, crops)):
        text = str(text).strip()
        if confidence >= PARSEQ_MIN_CONFIDENCE and is_valid_number(text):
            out[position] = text
    return out


@dataclass
class _Votes:
    counts: dict = field(default_factory=dict)  # normalized value -> count

    def add(self, value: str) -> None:
        self.counts[value] = self.counts.get(value, 0) + 1

    def resolved_counts(self, min_promote_votes: int, min_promote_ratio: float = 0.5) -> dict:
        """Counts with single-digit reads folded into the number they partially show.

        A crop that catches only the trailing digit of a two-digit number reads as that
        digit, so "7" is exactly what a partial view of "17" looks like. Those are
        corroborating observations of one number, not rival candidates, and leaving them
        to compete splits a player's evidence across both -- measured on FelixClaar,
        where a player wearing 17 accumulated {17: 4, 7: 4} and the tie suppressed any
        verdict.

        The asymmetry is measured, and it holds *at the confidence gate*, not
        absolutely. On the 323 labelled 1080p crops with the shipping reader, a
        two-digit jersey read as a single digit 3 times out of 237, and all 3 were the
        trailing digit -- but ungated there are 6 such reads, one of which is a leading
        digit. Every leading-digit case falls below `PARSEQ_MIN_CONFIDENCE`, so the gate
        is what makes one-sided folding safe, and lowering it would start admitting the
        case this rule assumes away. (An earlier note claimed 5 leading-digit reads in
        the labelled set; that was docTR's reader, not this one.)

        Note also how little this now does. Digit loss was severe on docTR, which is
        what the FelixClaar tie above shows; on the current reader it is 1.3% of
        two-digit reads. The rule is kept because it is conservative and its premise
        holds, not because it is still load-bearing.

        Folding is deliberately conservative:
          - only a 1-digit read folds, and only into a 2-digit read (never the reverse),
          - the longer reading needs both `min_promote_votes` of its own support AND
            `min_promote_ratio` of the support of the read it is absorbing, so a handful
            of misreads cannot capture a well-established shorter number. The absolute
            floor alone was not enough: measured on BHC-FAG, a player visibly wearing 7
            read {'7': 32, '77': 2} and two stray "77"s captured all 32, flipping a
            correct verdict to a wrong one.
          - if two equally-supported extensions exist ("17" and "27" both at 3), the
            short read cannot choose between them, so nothing folds.
        """
        merged = dict(self.counts)
        for short in [value for value in self.counts if len(value) == 1]:
            floor = max(min_promote_votes, min_promote_ratio * self.counts[short])
            extensions = sorted(
                (
                    (count, value) for value, count in self.counts.items()
                    if len(value) == 2
                    and value.endswith(short)
                    and count >= floor
                ),
                reverse=True,
            )
            if not extensions:
                continue
            if len(extensions) > 1 and extensions[0][0] == extensions[1][0]:
                continue
            target = extensions[0][1]
            merged[target] = merged.get(target, 0) + merged.pop(short, 0)
        return merged

    def best(self, min_promote_votes: int = 2, min_promote_ratio: float = 0.5):
        """(value, count, margin) for the leading valid number, or (None, 0, 0.0).

        Filtering happens *after* folding, not at `observe()`, and the ordering
        is load-bearing. A crop catching only the trailing digit of 10/20/40
        reads as "0", which is not itself a legal jersey number but is real
        corroboration for one -- measured on BHC-FAG, a stray "0" folded into
        p11's "40" and took it from 5 votes to 6. Dropping "0" on arrival would
        discard that. Barring it only from *winning* keeps the evidence and
        still prevents the verdict: on the same clip p3 read {"0": 13, ...} off
        a shorts logo the number detector mistook for a digit, and without this
        it reported 0 as that player's number.

        Invalid values are also excluded from `total`, so they cannot dilute a
        real value's margin -- a false detection is evidence about nothing.
        """
        counts = self.resolved_counts(min_promote_votes, min_promote_ratio)
        counts = {v: n for v, n in counts.items() if is_valid_number(v)}
        if not counts:
            return None, 0, 0.0
        total = sum(counts.values())
        value, count = max(counts.items(), key=lambda kv: kv[1])
        rest = sorted(counts.values(), reverse=True)[1:2]
        runner_up = rest[0] if rest else 0
        margin = (count - runner_up) / total
        return value, count, margin


# Reads off a different shirt before an identity disowns the number it inherited
# across a re-ID. One is a misread -- the readers in use are ~0.6 accurate on
# these crops -- and must not discard a well-supported tally; two is a pattern.
CONTRADICTIONS_BEFORE_DISOWNING = 2


class NumberVoter:
    """Per-identity jersey-number histogram that can change its mind.

    Unlike ConsecutiveValueTracker this never locks permanently -- necessary once
    re-ID or a hard reset can legitimately reassign what obj_id/player_id a
    physical player is tracked under, and a stale locked number would then be
    silently wrong for the rest of the clip.
    """

    def __init__(
        self, min_votes: int = 3, min_margin: float = 0.35, min_promote_votes: int = 2,
        min_promote_ratio: float = 0.5,
    ):
        self.min_votes = min_votes
        self.min_margin = min_margin
        self.min_promote_votes = min_promote_votes
        self.min_promote_ratio = min_promote_ratio
        self._votes: dict = {}
        # Values that have cleared both gates at least once. Consulted only when
        # the live tally has fallen back below them -- see `best`.
        self._settled: dict = {}
        # identity_id -> the identity that outvoted it for the same squad number.
        # Rebuilt from scratch by every `arbitrate` call, so it is a view of the
        # current evidence and never an accumulating penalty.
        self._suppressed: dict = {}
        # identity_id -> [verdict inherited across a re-ID, reads contradicting it].
        # See `suspend`.
        self._unconfirmed: dict = {}

    def observe(self, identity_id: int, raw_value: str) -> None:
        value = self._normalize(raw_value)
        if value is None:
            return
        self._votes.setdefault(identity_id, _Votes()).add(value)
        qualified, _count, _margin = self._qualified(identity_id)
        if qualified is not None:
            self._settled[identity_id] = qualified
        pending = self._unconfirmed.get(identity_id)
        if pending is None:
            return
        inherited = pending[0]
        # The same fold rule `resolved_counts` uses: a partial view of "15"
        # reads "5", so a single digit backs the number it could have come from.
        if value == inherited or (len(value) == 1 and inherited.endswith(value)):
            self._unconfirmed.pop(identity_id, None)
            return
        pending[1] += 1
        if pending[1] < CONTRADICTIONS_BEFORE_DISOWNING:
            return
        # This identity is now reading a different shirt, so the inherited
        # tally describes somebody else and has to go -- leaving it in place
        # would sit in the denominator forever, and a fresh number would need
        # ~25 corroborating reads to clear `min_margin` against it. Only the
        # contradicting reads survive, as the new fragment's own first evidence.
        self._unconfirmed.pop(identity_id, None)
        self._settled.pop(identity_id, None)
        self._votes[identity_id] = _Votes()
        for _ in range(pending[1]):
            self._votes[identity_id].add(value)

    def _qualified(self, identity_id: int):
        """(value, count, margin) where value is set only if both gates pass now."""
        v = self._votes.get(identity_id)
        if v is None:
            return None, 0, 0.0
        value, count, margin = v.best(self.min_promote_votes, self.min_promote_ratio)
        if value is None or count < self.min_votes or margin < self.min_margin:
            return None, count, margin
        return value, count, margin

    @staticmethod
    def _normalize(raw_value):
        s = (raw_value or "").strip()
        return s if s.isdigit() and len(s) <= 2 else None

    def _verdict(self, identity_id: int):
        """(number, votes, margin). None until min_votes/min_margin are first met.

        Once a value has qualified, it is held until a *different* value
        qualifies -- the verdict does not evaporate merely because contrary
        reads dragged the margin back under the floor. Measured on BHC-FAG:
        p13 (jersey 25) reached `{'25': 3}` at margin 1.0, then four partial
        "2" reads pulled it to 0.143 and the label reverted to `P13` before
        recovering, and p3 flickered the same way on "53". Recomputing from
        scratch every frame makes resolution non-monotonic and the overlay
        flicker visible.

        This is hysteresis, not the permanent lock `ConsecutiveValueTracker`
        applies: a rival that clears both gates replaces the held value, so
        re-ID and genuine corrections still work.

        Holding a verdict is only safe if the bar to set one is high enough
        that a short run of correlated misreads cannot set it, and the useful
        bar is `min_margin`, not `min_votes`. Both bad commits measured here
        happened on three votes -- but so did a correct one, and margin
        separates them: BHC-FAG p3 (visually confirmed jersey 53) committed at
        {'53': 3}, margin 1.000, while EasyOCR p2 (jersey 22) committed at
        {'92': 3, '22': 2}, margin 0.200, contested from the first read.
        Raising `min_votes` to 5 blocked both and cost p3 its correct answer,
        since every later read of that player was shorts-logo noise; raising
        `min_margin` to 0.35 blocks only the contested one.
        """
        v = self._votes.get(identity_id)
        if v is None:
            return None, 0, 0.0
        value, count, margin = self._qualified(identity_id)
        if value is not None:
            return value, count, margin
        held = self._settled.get(identity_id)
        if held is not None:
            counts = v.resolved_counts(self.min_promote_votes, self.min_promote_ratio)
            return held, counts.get(held, 0), margin
        return None, count, margin

    def suspend(self, identity_id: int):
        """Stop asserting this identity's number until a fresh read backs it.

        A verdict is evidence about a *person*, but it is filed against an
        identity -- and re-ID moves an identity onto whoever it believes has
        reappeared, which within a team is right about half the time. Measured
        on the 60s Melsungen clip, p6 settled on 15 from nine reads over frames
        65-110, was revived onto a different player at frame 560, then read
        20, 29 and 2 off that player's shirt while still labelled 15. The
        hysteresis in `_verdict` is what makes that stable: it holds a value
        until a *rival* qualifies, and one or two contrary reads never do.

        Votes are kept rather than cleared, so a correct revival is vouched for
        by its first agreeing read, while a wrong one never asserts the number
        it inherited. Call this on every re-ID revival.
        """
        held = self._verdict(identity_id)[0]
        if held is not None:
            self._unconfirmed[identity_id] = [held, 0]
        return held

    def claim(self, identity_id: int):
        """`_verdict`, unless it was inherited across a re-ID and never vouched.

        What this identity may assert against *others* -- who owns a squad
        number, and which identities are the same player. Distinct from `best`,
        which additionally withholds a claim that lost such a contest.
        """
        if identity_id in self._unconfirmed:
            _value, count, margin = self._verdict(identity_id)
            return None, count, margin
        return self._verdict(identity_id)

    def best(self, identity_id: int):
        """What may be displayed: a `claim` that no stronger one has beaten.

        A suppressed identity reports no number rather than a wrong one, which
        is the project's standing rule: abstaining beats injecting a confident
        wrong observation. See `arbitrate`.
        """
        if identity_id in self._suppressed:
            _value, count, margin = self._verdict(identity_id)
            return None, count, margin
        return self.claim(identity_id)

    def arbitrate(self, team_by_identity: dict) -> dict:
        """One squad number, one player per team: withhold the weaker claims.

        A number is unique within a team, so two identities holding the same one
        at the same time is proof that at least one of them is wrong -- and the
        pipeline could previously state it anyway, because votes are counted per
        identity in isolation with nothing comparing them. Measured on the 60s
        Melsungen clip, three identities resolved to `25` and two pairs of them
        were read in the *same frame*, so they cannot be one fragmented player.

        This does not repair the underlying identity error: the loser is still
        whoever the tracker mistakenly grabbed. It stops the run asserting
        something that cannot be true, and it is reversible -- the ranking is
        recomputed from current vote counts on every call, so a loser that later
        overtakes the holder takes the number back.

        `team_by_identity` should carry the identities in play; pass live ones to
        arbitrate what is on screen. Returns {suppressed: identity that kept it}.
        """
        claims: dict = {}
        for identity_id, team_id in team_by_identity.items():
            value, count, margin = self.claim(identity_id)
            if value is not None:
                claims.setdefault((team_id, value), []).append(
                    (count, margin, identity_id)
                )
        self._suppressed = {}
        for holders in claims.values():
            if len(holders) < 2:
                continue
            # Votes first, then margin; identity_id last only so a tie resolves
            # the same way twice rather than flickering between frames.
            holders.sort(key=lambda h: (-h[0], -h[1], h[2]))
            keeper = holders[0][2]
            for _count, _margin, loser in holders[1:]:
                self._suppressed[loser] = keeper
        return dict(self._suppressed)

    def reset(self, identity_id: int) -> None:
        self._votes.pop(identity_id, None)
        self._settled.pop(identity_id, None)
        self._suppressed.pop(identity_id, None)
        self._unconfirmed.pop(identity_id, None)

    def merge(self, from_id: int, into_id: int) -> None:
        """Fold one identity's votes into another's -- call this on a re-ID hit
        so accumulated evidence isn't discarded when a tracker-level id changes."""
        src = self._votes.pop(from_id, None)
        self._settled.pop(from_id, None)
        self._suppressed.pop(from_id, None)
        self._unconfirmed.pop(from_id, None)
        if src is None:
            return
        dst = self._votes.setdefault(into_id, _Votes())
        for value, count in src.counts.items():
            dst.counts[value] = dst.counts.get(value, 0) + count
        # The merged tally is new evidence: re-derive rather than inherit either
        # side's held value, so a merge cannot smuggle in a verdict the combined
        # counts do not support.
        self._settled.pop(into_id, None)
        qualified, _c, _m = self._qualified(into_id)
        if qualified is not None:
            self._settled[into_id] = qualified


# Event types that end a segment. A `reid` revival moves an identity onto
# whoever the tracker believes has reappeared, and a `suspected_id_switch` says
# it may already have moved -- either way the frames after are not guaranteed to
# be the same person as the frames before. `team_switch` and `link` are not
# breaks: a switch revises a label on one continuous track, and a link declares
# two identities were the same person all along, which joins evidence rather
# than cutting it.
IDENTITY_BREAK_EVENTS = ("reid", "suspected_id_switch")


def segment_breaks(events) -> dict:
    """{player_id: sorted break frames} from a run's `identity_events`."""
    breaks: dict = {}
    for event in events:
        if event.get("type") not in IDENTITY_BREAK_EVENTS:
            continue
        breaks.setdefault(event["player_id"], set()).add(event["frame"])
    return {pid: sorted(frames) for pid, frames in breaks.items()}


@dataclass(frozen=True)
class Segment:
    """One identity's span between identity breaks -- and so one person.

    `timeline` is the causal record: `((frame, value), ...)`, one entry each
    time the label the render drew for this player changed within this span. It
    comes from replaying the run's real voting, so it is what the overlay
    actually showed, not a reconstruction of what it could have shown.

    A segment whose label never changed was right about that number for its
    whole span, so the number may be stated over the whole span. One that
    changed revised itself, and its early frames' own evidence said something
    else -- those frames keep what they were shown.
    """
    player_id: int
    start: int
    end: int                      # inclusive
    live: tuple                   # live frames within [start, end]
    timeline: tuple               # ((frame, value), ...) causal label changes
    votes: int                    # final tally, for reporting
    margin: float

    def _causal_at(self, frame: int) -> str | None:
        """The label the render drew on `frame`, `None` where it drew none."""
        value = None
        for at, candidate in self.timeline:
            if at > frame:
                break
            value = candidate
        return value

    @property
    def _values(self) -> list:
        return [value for _, value in self.timeline if value is not None]

    @property
    def verdict(self) -> str | None:
        """The number this segment settles on by its end."""
        values = self._values
        return values[-1] if values else None

    @property
    def resolved_at(self) -> int | None:
        """Frame the render first drew a number for this segment."""
        return next((f for f, value in self.timeline if value is not None), None)

    @property
    def stable(self) -> bool:
        """Whether one number, and only one, was ever stated in this span.

        A `None` in the timeline is arbitration withholding a duplicate claim,
        not a revision, so it does not count against stability -- the segment
        never said anything else.
        """
        return len(set(self._values)) <= 1

    @property
    def labelled(self) -> tuple:
        """Live frames the render already drew a number on."""
        return tuple(f for f in self.live if self._causal_at(f) is not None)

    @property
    def backfill(self) -> tuple:
        """Live frames this verdict describes but no label ever reached.

        Empty when the label changed mid-segment. Painting the final answer over
        frames whose own evidence said something else would overwrite what they
        actually showed -- measured at 1 segment in 14 on the 60s Melsungen
        clip, where p20 went 6 -> 15, and worth the 284 frames it costs.
        """
        if not self.stable or self.resolved_at is None:
            return ()
        return tuple(f for f in self.live if f < self.resolved_at)

    def value_at(self, frame: int) -> str | None:
        """What may be drawn on `frame`.

        At or after `resolved_at` this is exactly what the render drew,
        withheld duplicates included. Before it, a stable segment extends its
        verdict backwards -- the backfill -- and a revised one draws nothing.
        """
        if self.resolved_at is None:
            return None
        if frame < self.resolved_at:
            return self.verdict if self.stable else None
        return self._causal_at(frame)


def causal_labels(reads, events, live_frames, teams=None,
                  ocr_every: int = OCR_EVERY_N_FRAMES) -> dict:
    """{player_id: ((frame, label), ...)} as the causal render drew them.

    Replays the render's own labelling loop rather than re-deriving one, because
    the two are not the same and the difference is not small. Three behaviours
    have to come along or a redraw silently drops labels the render showed:

      - `suspend` on a re-ID revival keeps the accumulated votes and merely
        withholds the claim, so a correct revival re-asserts on its first
        agreeing read. Starting a fresh tally at each break instead costs
        `min_votes` reads every time -- measured at 765 player-frames on the 60s
        Melsungen clip (p1 and p12 for 475 frames each, p16 for 10), which is a
        regression dressed up as caution.
      - arbitration runs every `ocr_every` frames over every identity, live and
        retired, so a withheld duplicate stays withheld between OCR frames.
      - a `suspected_id_switch` breaks a *segment* but does not suspend, exactly
        as `NumberIdentityResolver.begin_frame` has it.

    Only entries where the label changes are recorded; read them with
    `Segment.value_at`.
    """
    reads_by_frame: dict = {}
    for row in reads:
        reads_by_frame.setdefault(int(row["frame"]), []).append(
            (int(row["player_id"]), row["value"])
        )
    revivals: dict = {}
    for event in events:
        if event.get("type") == "reid":
            revivals.setdefault(int(event["frame"]), []).append(int(event["player_id"]))

    voter = NumberVoter()
    timelines: dict = {}
    held: dict = {}
    # Revival frames join the walk even if nobody is on screen then: a
    # `suspend` that is never applied leaves the pre-break verdict asserting
    # itself into the new segment. On a full clip somebody is always live so it
    # never shows, which is exactly why it needs stating.
    walk = {f for frames in live_frames.values() for f in frames} | set(revivals)
    for frame in sorted(walk):
        for player_id in revivals.get(frame, ()):
            voter.suspend(player_id)
        for player_id, value in reads_by_frame.get(frame, ()):
            voter.observe(player_id, value)
        if teams and frame % ocr_every == 0:
            voter.arbitrate(teams)
        for player_id, frames in live_frames.items():
            if frame not in frames:
                continue
            label = voter.best(player_id)[0]
            if label != held.get(player_id):
                held[player_id] = label
                timelines.setdefault(player_id, []).append((frame, label))
    return {pid: tuple(entries) for pid, entries in timelines.items()}


def segment_verdicts(reads, events, live_frames, teams=None,
                     ocr_every: int = OCR_EVERY_N_FRAMES) -> list:
    """Per-segment verdicts, and which frames each one may backfill.

    A number is only displayed from the frame its third read lands, so the
    frames before it carry `P<id>` even though the same evidence explains them.
    A verdict earned late still describes the start of the span it was earned
    in -- but only that span. Re-ID moves an identity onto a different person,
    so a verdict must never reach across a break: on the 60s Melsungen clip p6
    is 15 until frame 560 and somebody else after, and a whole-identity
    backfill would paint 15 over the second person too.

    Segments therefore bound how far back a label may reach; they do not
    partition the evidence. The labels themselves come from `causal_labels`,
    which replays the run's actual voting -- so at and after `resolved_at` a
    redraw reproduces the render exactly, and only the frames before it change.

    Args:
        reads: rows of `{frame, player_id, value}`, a run's `number_reads`.
        events: a run's `identity_events`.
        live_frames: {player_id: frames the identity was on screen}.
        teams: {player_id: team_id} for arbitration; omitted, none is applied.
        ocr_every: arbitration cadence, matching the render's `--ocr-every`.

    Returns segments in (player_id, start) order, including those that never
    resolved -- a caller measuring coverage needs the denominator too.
    """
    breaks = segment_breaks(events)
    timelines = causal_labels(reads, events, live_frames, teams, ocr_every)
    counts: dict = {}
    voter = NumberVoter()
    for row in sorted(reads, key=lambda r: r["frame"]):
        voter.observe(int(row["player_id"]), row["value"])
    for player_id in live_frames:
        counts[player_id] = voter.best(player_id)[1:]

    segments = []
    for player_id in sorted(live_frames):
        live = sorted(set(live_frames[player_id]))
        if not live:
            continue
        # A break is the first frame of the segment it opens. Breaks outside the
        # live span would only open empty segments, so they are dropped here
        # rather than filtered out again after.
        cuts = sorted({
            frame for frame in breaks.get(player_id, ())
            if live[0] < frame <= live[-1]
        })
        bounds = [live[0], *cuts, live[-1] + 1]
        timeline = timelines.get(player_id, ())
        votes, margin = counts.get(player_id, (0, 0.0))
        for start, stop in zip(bounds, bounds[1:]):
            span = tuple(f for f in live if start <= f < stop)
            if not span:
                continue
            within = tuple(
                (frame, label) for frame, label in timeline if start <= frame < stop
            )
            # A `suspected_id_switch` breaks the segment without suspending the
            # voter, so the label simply continues and records no new entry. The
            # new segment has to be told what was already on screen, or it reads
            # as unresolved and drops a label the render was drawing.
            #
            # Anchored on the first *live* frame, never on the break frame: an
            # identity that is off screen when it breaks -- p1 breaks at 540 and
            # returns at 541 -- otherwise gets the previous person's label
            # planted at the boundary, which both back-dates `resolved_at` to the
            # segment start (erasing its whole backfill) and carries a verdict
            # across the break that segmentation exists to stop.
            if not within or within[0][0] > span[0]:
                carried = None
                for frame, label in timeline:
                    if frame >= span[0]:
                        break
                    carried = label
                if carried is not None:
                    within = ((span[0], carried), *within)
            segments.append(Segment(
                player_id=player_id,
                start=span[0],
                end=span[-1],
                live=span,
                timeline=within,
                votes=votes,
                margin=margin,
            ))
    return segments
