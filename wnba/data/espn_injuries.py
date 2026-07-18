"""Real, current league-wide injury report from ESPN. Confirmed live (no
auth): status (Out/Questionable/Day-To-Day/...), injury type, and a
human-written comment per player, refreshed continuously.

Fed into player-prop projections as of the injury-aware opponent adjustment
in model/player_model.py's compute_out_player_adjustment -- but only the
`status == "Out"` rows, and only for a player with enough of a rotation
track record to matter (see PLAYER_INJURY_MIN_MINUTES/MIN_GAMES in
config.py). "Questionable"/"Day-To-Day" aren't reliable enough to condition
on before the game actually happens, so those stay display-only, same as
before. player_id is parsed out of the athlete's "playercard" link (the raw
injuries payload has no top-level athlete id field -- its "id" is the
injury record's own id) so it can be joined against player_box/availability
data by id, not name.
"""
from __future__ import annotations

import re

import pandas as pd
import requests

from wnba.config import ESPN_INJURIES_URL

_PLAYER_ID_RE = re.compile(r"/id/(\d+)/")


def _extract_player_id(athlete: dict) -> str | None:
    for link in athlete.get("links", []):
        if "playercard" in link.get("rel", []):
            match = _PLAYER_ID_RE.search(link.get("href", ""))
            if match:
                return match.group(1)
    return None


def fetch_injuries() -> pd.DataFrame:
    resp = requests.get(ESPN_INJURIES_URL, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for team in data.get("injuries", []):
        for injury in team.get("injuries", []):
            athlete = injury.get("athlete", {})
            rows.append({
                "team": team.get("displayName"),
                "player_id": _extract_player_id(athlete),
                "player_name": athlete.get("displayName"),
                "position": athlete.get("position", {}).get("abbreviation"),
                "status": injury.get("status"),
                "date": injury.get("date"),
                "comment": injury.get("shortComment") or injury.get("longComment"),
            })
    return pd.DataFrame(rows)


def team_injuries(team: str, injuries: pd.DataFrame | None = None) -> pd.DataFrame:
    injuries = injuries if injuries is not None else fetch_injuries()
    return injuries[injuries["team"] == team].reset_index(drop=True)
