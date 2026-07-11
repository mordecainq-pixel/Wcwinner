"""Real, current league-wide injury report from ESPN. Confirmed live (no
auth): status (Out/Questionable/Day-To-Day/...), injury type, and a
human-written comment per player, refreshed continuously.

Deliberately still not fed into the model's quantitative predictions --
that would mean deciding how many points a specific absence is worth, which
needs real validation (e.g. does removing a player's recent per-minute
production and redistributing their minutes actually improve backtested
accuracy?) before it should move a probability. This module only surfaces
the report for display, same as wcwinner's manual home/away strength
multiplier leaves the *decision* of how much an absence matters to you --
the difference here is just that the report itself is fetched automatically
instead of typed in by hand, because for the WNBA a good free source
actually exists.
"""
from __future__ import annotations

import pandas as pd
import requests

from wnba.config import ESPN_INJURIES_URL


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
