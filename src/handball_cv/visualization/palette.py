"""How a tracked player is coloured, shared by every renderer and the result bundle.

One palette, indexed by a player's team state. Team A/B come first so a
roster-less clip still reads like the notebook's two-team overlay. The two
non-team states are kept apart on purpose: a brand-new player still building
evidence is just bootstrapping, while an established one whose team is under
active opposition is the interesting failure, and should stand out.
"""
from __future__ import annotations

TEAM_A, TEAM_B, NEW_UNCERTAIN, CONTESTED, GOALKEEPER = 0, 1, 2, 3, 4
PALETTE_HEX = [
    "#00BEFF",  # A
    "#FF7800",  # B
    "#FFDC00",  # new / never had a qualified read
    "#DC0000",  # established but actively contested
    "#B4B4B4",  # goalkeeper
]
LEGEND = [
    (TEAM_A, "Team A"),
    (TEAM_B, "Team B"),
    (NEW_UNCERTAIN, "New / uncertain"),
    (CONTESTED, "Contested"),
    (GOALKEEPER, "GK"),
]

# The team state each palette index stands for, as the result bundle records it.
# A goalkeeper casts no team vote, so it has no team state.
TEAM_STATE = {
    TEAM_A: "stable",
    TEAM_B: "stable",
    NEW_UNCERTAIN: "provisional",
    CONTESTED: "contested",
    GOALKEEPER: None,
}


def color_index(player) -> int:
    """Palette entry for a live `PlayerRecord`."""
    # Imported here, not at the top: the team model brings umap and torch
    # (~4 s), and only the live paths that already hold one call this. A
    # finished bundle's render imports the palette and nothing else of it.
    from handball_cv.teams.model import MIN_STABLE_TEAM_CONFIDENCE

    if player.is_goalkeeper:
        return GOALKEEPER
    if player.team_is_provisional:
        return NEW_UNCERTAIN
    if player.team_confidence < MIN_STABLE_TEAM_CONFIDENCE:
        return CONTESTED
    return TEAM_A if player.voted_team_id == 0 else TEAM_B


def color_index_for(team: int, team_state: str | None, is_goalkeeper: bool) -> int:
    """Palette entry from the bundle's recorded columns -- the inverse of `TEAM_STATE`."""
    if is_goalkeeper:
        return GOALKEEPER
    if team_state == "provisional":
        return NEW_UNCERTAIN
    if team_state == "contested":
        return CONTESTED
    return TEAM_A if team == 0 else TEAM_B
