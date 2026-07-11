"""Today's (or any given day's) WNBA schedule, including games that haven't
tipped off yet -- unlike espn_ingest's historical pipeline, which only keeps
*completed* games so it never has to worry about a fixture's score changing
later. Also resolves a team's current active roster from recent box scores,
since ESPN doesn't expose a clean standalone roster endpoint via this API.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from wnba.config import ESPN_BASE_URL


def todays_games(on_date: str | None = None) -> list[dict]:
    """Every game scheduled for `on_date` (default: today), whatever its
    status -- scheduled, in-progress, or final.
    """
    day = on_date.replace("-", "") if on_date else date.today().strftime("%Y%m%d")
    resp = requests.get(f"{ESPN_BASE_URL}/scoreboard", params={"dates": day}, timeout=20)
    resp.raise_for_status()
    events = resp.json().get("events", [])

    games = []
    for event in events:
        comp = event["competitions"][0]
        competitors = {c["homeAway"]: c for c in comp["competitors"]}
        if "home" not in competitors or "away" not in competitors:
            continue
        home, away = competitors["home"], competitors["away"]
        games.append({
            "event_id": event.get("id"),
            "date": event["date"][:10],
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "status": comp["status"]["type"]["description"],
            "completed": bool(comp["status"]["type"]["completed"]),
            "home_score": int(home["score"]) if home.get("score") not in (None, "") else None,
            "away_score": int(away["score"]) if away.get("score") not in (None, "") else None,
        })
    return games


def team_roster(team: str, player_box: pd.DataFrame, n_recent_games: int = 5) -> pd.DataFrame:
    """A team's likely active players: everyone who appeared in that team's
    last `n_recent_games`, most recently used first. Best-effort proxy for a
    current roster -- there's no clean "who's on the roster today" endpoint,
    and this at least excludes players who were traded/waived/retired a
    while back.
    """
    team_games = player_box[player_box["team"] == team]
    if team_games.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "position", "games_played", "last_played"])

    recent_dates = sorted(team_games["date"].unique())[-n_recent_games:]
    recent = team_games[team_games["date"].isin(recent_dates)]

    agg = recent.groupby(["player_id", "player_name", "position"]).agg(
        games_played=("date", "count"), last_played=("date", "max"),
    ).reset_index()
    return agg.sort_values("last_played", ascending=False).reset_index(drop=True)
