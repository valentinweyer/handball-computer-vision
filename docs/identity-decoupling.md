# Decoupling tracker error from team error (2026-08-24)

Full record of the two identity/team decoupling fixes made on 2026-08-24, moved
out of `CLAUDE.md` on 2026-08-28 to keep the always-loaded project instructions
small. `CLAUDE.md` retains the constants and invariants; everything below is the
discovery narrative, verification evidence, and measurements behind them.

### Decoupling tracker error from team error (2026-08-24)

The reversible-switch mechanism above was built to correct *classifier* error (an unlucky initial crop). It was found to also silently absorb *tracker* error: when McByte swaps two crossing players under one `tracker_id`, the resulting observations are correct about the pixels and wrong about the identity, and look exactly like a legitimate correction. The old mechanism reset `team_votes` to just the switch-triggering weight, so a long-lived, well-evidenced player who flipped this way still read as "stable" (confidence ~0.78, above `MIN_STABLE_TEAM_CONFIDENCE = 0.70`) immediately after flipping — and its new, wrong team label then hard-excluded the correct re-ID candidates. Tracking error became team error became re-ID error, the exact circular failure the architecture is meant to prevent, arriving by a different route.

Fix, in `notebooks/identity_manager.py`:

- **Evidence decays instead of accumulating without bound.** `TEAM_EVIDENCE_DECAY = 0.85`, applied to `team_votes` before each new observation is added. An unbounded accumulator saturates `team_confidence` near 1.0 within a few hundred frames and can never again express doubt. Decay bounds the mass at roughly `weight / (1 - decay)` (≈6 at 0.85) and caps confidence at ≈0.93, so a *contested* label — not just a flipped one — drops below the stable gate and stops vetoing re-ID candidates before the label itself has switched.
- **Prior evidence is kept, not discarded, on a switch.** The old `self.team_votes = {team_id: pending_weight}` reset is gone; the decayed history stays.
- **A flip is classified by how settled the label was.** If `team_confidence >= MIN_STABLE_TEAM_CONFIDENCE` and `qualified_observations >= TEAM_SETTLED_OBSERVATIONS` (5) at the moment opposition began, the flip logs as `suspected_id_switch` instead of `team_switch` — diagnostic only, no behavioral branch, but it makes the tracker-error rate measurable.
- **A label with zero qualified observations behind it (`team_is_provisional`) yields to the first qualified read immediately** (`record_team_vote` outcome `"adopt"`), rather than needing three opposing observations like a real switch. The one-frame creation guess had no evidentiary claim to that inertia.
- **The re-ID team veto now requires evidence on both sides**: the incoming observation must itself be qualified (shared `is_qualified(confidence, quality)` predicate — the vote gate and the re-ID gate used to compare different quantities against the same threshold, `confidence*quality` vs `confidence`, making the re-ID gate silently stricter), and the candidate's own label must be non-provisional and above `MIN_STABLE_TEAM_CONFIDENCE`.
- **Goalkeeper status is now a running majority (`Player.is_goalkeeper`), not frozen at creation.** It only vetoes re-ID once `goalkeeper_settled` (`GOALKEEPER_EVIDENCE_MIN = 5` observations, `GOALKEEPER_EVIDENCE_MAJORITY = 0.70`) — a single flickered detector class at creation used to permanently fork an identity from its own fragments across the strictest gate in the module.

New constants: `TEAM_EVIDENCE_DECAY = 0.85`, `TEAM_SETTLED_OBSERVATIONS = 5`, `GOALKEEPER_EVIDENCE_MIN = 5`, `GOALKEEPER_EVIDENCE_MAJORITY = 0.70`.

`IdentityManager.summary()` now reports `team_switches` and `suspected_id_switches` separately. `render_mcbyte_team_correction.py`'s on-frame header and result JSON follow the same split; the JSON's `team_switches` key was renamed to `label_changes_total` (the two counts' sum) so it isn't silently overwritten by `summary()`'s `**` merge.

Verified against the Felix and Han-Ber clips (`outputs/team_correction_mcbyte/FelixClaar_idswitch_h264.mp4`, `Han-Ber4_idswitch_h264.mp4`):

- Felix: 11 total label changes (same as the prior baseline), now split 5 `team_switch` / 6 `suspected_id_switch`. `reid_hits` (4), `fragmented_players` (4), `max_fragments_per_player` (2) all unchanged.
- Han-Ber: 5 total label changes (down from the prior baseline's 6 — one previously-provisional flip now resolves via `"adopt"` and is no longer counted as a switch), split 2 `team_switch` / 3 `suspected_id_switch`. `reid_hits` (2), `fragmented_players` (1), `max_fragments_per_player` (3) all unchanged.
- Spot-checked frames around two `suspected_id_switch` events (Felix frame 176, Han-Ber frame 175) by extracting stills from the rendered video: both land in dense, tightly-clustered/occluded moments, consistent with the tracker-swap hypothesis rather than a color misclassification.
- No regression in re-ID or fragmentation counts on either clip. Per-frame team accuracy was not expected to move and there is no evidence it did — this change targets identity/re-ID robustness, not classifier accuracy.
- 10 new tests added to `tests/test_team_model.py` (`TrackerErrorIsolationTests`, `ProvisionalTeamLabelTests`, `QualifiedObservationTests`, `GoalkeeperRoleTests`); full suite is 38 tests, all passing.

`0.85` for `TEAM_EVIDENCE_DECAY` is a reasoned estimate (it sets how many observations a legitimate switch needs to fully re-stabilize, ≈10 observations ≈ 50 frames at the current sampling cadence), not a value tuned against labeled data — there is no labeled ID-switch dataset to tune it against. Revisit if `suspected_id_switch` rates look wrong on new videos.

### Decay-on-noise bug and the two-tone "uncertain" display (2026-08-24, same day)

While reviewing rendered output, a second, narrower bug surfaced in the decay mechanism above: `Player.record_team_vote` applied decay unconditionally on every call, but `IdentityManager.update()`'s call-site gate (`confidence >= MIN_TEAM_VOTE_CONFIDENCE and quality > 0`) is looser than `is_qualified()` (`quality >= TEAM_SWITCH_MIN_QUALITY`). A read that passed the call site but failed `is_qualified` — a routine low-quality observation from partial occlusion or a small crop — still decayed 15% off the accumulated evidence every time, while contributing only a tiny counterbalancing weight. In a controlled repro, ~20 such reads alone dragged a settled player from `team_confidence` 0.93 down to 0.65 (below the display/stability threshold) with zero genuine opposition.

Fix: the `is_qualified(confidence, quality)` check now gates entry to `record_team_vote` — decay and vote weight are only applied for a genuinely qualified observation; an unqualified read is a true no-op for `team_votes` (it still increments the diagnostic `team_observations` counter). Confirmed by re-running the repro (confidence held flat through 25 subsequent weak reads) and by 3 new tests in `TrackerErrorIsolationTests`.

Measuring the real-world impact on Felix and Han-Ber (via an instrumented run comparing buggy vs. fixed decay, no video written) showed this bug was **not** the dominant driver of "uncertain" display time on these two clips: established-player uncertain-frame rates were nearly identical either way (Felix 17.1% buggy vs. 17.7% fixed; Han-Ber 13.6% vs. 13.4%), and logged switch/suspected-id-switch counts were unchanged (11 and 5 respectively). Most of the observed uncertainty on these clips is genuine, recurring qualified disagreement during frequent crossings/overlaps — not a noise artifact. The fix is still correct and worth keeping (it's a real conceptual bug), but don't expect it alone to visibly quiet these two clips.

Separately, `render_team_overlay.py` and `render_mcbyte_team_correction.py` now distinguish two different "not a stable team color" states instead of one shared yellow:

- **New/uncertain** (`(0, 220, 255)`, unchanged color): `player.team_is_provisional` — never had a qualified observation yet. Expected and uninteresting; every player starts here.
- **Contested** (`(0, 0, 220)`, new): established (non-provisional) but `team_confidence < MIN_STABLE_TEAM_CONFIDENCE` — under genuine active opposition right now. This is the signal worth noticing: often a crossing or the early stage of a suspected tracker swap. It is intentionally still shown, not hidden, even though it makes established players occasionally flicker red — muting it would hide exactly the failure mode this whole session's work was built to surface.

Team/identity changes were already gated on qualification before this change (a switch needs 3 qualified opposing reads; re-ID already refuses a provisional or contested candidate) — the two-tone split is a display-only change, not a behavioral one.
