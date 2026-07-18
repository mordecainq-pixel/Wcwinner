"""Per-player box scores from ESPN's summary endpoint, keyed off the game
`event_id`s already collected by espn_ingest.py's team-level pipeline.

One request per game (confirmed live, both current and back to at least
2015): boxscore.players is a two-element list (one per team), each with a
single statistics group whose `keys` (machine-readable) and matching `stats`
(parallel string list, `[]` for a DNP player) give exactly the categories
props need: minutes, points, FG/3PT/FT splits, rebounds, assists, turnovers,
steals, blocks, offensive/defensive rebound split, fouls, +/-.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from wnba.config import DATA_RAW_DIR, ESPN_BASE_URL, PLAYER_BOX_MIN_REQUEST_INTERVAL, PLAYER_LOOKBACK_YEARS
from wnba.data.espn_ingest import load_results

_PLAYER_BOX_FILE = DATA_RAW_DIR / "player_boxscores.csv"
_PLAYER_AVAILABILITY_FILE = DATA_RAW_DIR / "player_availability.csv"

# ESPN box score `keys` -> our column names.
_STAT_KEY_MAP = {
    "minutes": "minutes",
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "turnovers": "turnovers",
    "steals": "steals",
    "blocks": "blocks",
    "offensiveRebounds": "oreb",
    "defensiveRebounds": "dreb",
    "fouls": "fouls",
    "plusMinus": "plus_minus",
}
# made-attempted split stats need their own parsing (e.g. "4-10" -> 4, 10)
_SPLIT_STAT_KEY_MAP = {
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
}


def _fetch_boxscore(event_id: str) -> dict:
    resp = requests.get(f"{ESPN_BASE_URL}/summary", params={"event": event_id}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _to_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _parse_athlete_stats(keys: list[str], stats: list[str]) -> dict | None:
    if not stats:
        return None  # did-not-play / inactive: ESPN returns an empty stats list
    row = {}
    for key, value in zip(keys, stats):
        if key in _STAT_KEY_MAP:
            row[_STAT_KEY_MAP[key]] = _to_float(value)
        elif key in _SPLIT_STAT_KEY_MAP:
            made_col, att_col = _SPLIT_STAT_KEY_MAP[key]
            made, _, att = value.partition("-")
            row[made_col] = _to_float(made)
            row[att_col] = _to_float(att)
    return row


def _parse_boxscore(summary: dict, event_id: str, date: str, season: int, is_playoff: bool) -> list[dict]:
    team_groups = summary.get("boxscore", {}).get("players", [])
    if len(team_groups) != 2:
        return []  # malformed/incomplete summary (e.g. postponed/forfeited game)

    rows = []
    teams_by_order = sorted(team_groups, key=lambda g: g.get("displayOrder", 0))
    for i, group in enumerate(teams_by_order):
        team_name = group.get("team", {}).get("displayName")
        opponent_name = teams_by_order[1 - i].get("team", {}).get("displayName")
        stat_blocks = group.get("statistics", [])
        if not stat_blocks:
            continue
        keys = stat_blocks[0].get("keys", [])
        for athlete in stat_blocks[0].get("athletes", []):
            parsed = _parse_athlete_stats(keys, athlete.get("stats", []))
            if parsed is None:
                continue
            info = athlete.get("athlete", {})
            rows.append({
                "event_id": event_id,
                "date": date,
                "season": season,
                "is_playoff": is_playoff,
                "team": team_name,
                "opponent": opponent_name,
                "player_id": info.get("id"),
                "player_name": info.get("displayName"),
                "position": info.get("position", {}).get("abbreviation"),
                "starter": bool(athlete.get("starter", False)),
                **parsed,
            })
    return rows


def _parse_availability(summary: dict, event_id: str, date: str, season: int, is_playoff: bool) -> list[dict]:
    """Every ROSTERED athlete for both teams in this game, played or not --
    the DNP/inactive players `_parse_boxscore` discards entirely. Reuses the
    same fetched summary as `_parse_boxscore` (no extra requests). This is
    what lets a currently-injured player's HISTORICAL absences be identified
    later (see model/player_model.py's compute_out_player_adjustment): a
    game where `played` is False for a given player_id is a confirmed DNP,
    same signal a live injury report gives beforehand, just after the fact.
    """
    team_groups = summary.get("boxscore", {}).get("players", [])
    if len(team_groups) != 2:
        return []

    rows = []
    teams_by_order = sorted(team_groups, key=lambda g: g.get("displayOrder", 0))
    for i, group in enumerate(teams_by_order):
        team_name = group.get("team", {}).get("displayName")
        opponent_name = teams_by_order[1 - i].get("team", {}).get("displayName")
        stat_blocks = group.get("statistics", [])
        if not stat_blocks:
            continue
        for athlete in stat_blocks[0].get("athletes", []):
            info = athlete.get("athlete", {})
            rows.append({
                "event_id": event_id,
                "date": date,
                "season": season,
                "is_playoff": is_playoff,
                "team": team_name,
                "opponent": opponent_name,
                "player_id": info.get("id"),
                "player_name": info.get("displayName"),
                "played": bool(athlete.get("stats")),
            })
    return rows


def refresh(lookback_years: int = PLAYER_LOOKBACK_YEARS) -> pd.DataFrame:
    """Fetches box scores for every completed game in the last
    `lookback_years` (from the already-ingested team results, so run
    espn_ingest.refresh() first). One request per game -- slow (~1/sec) but
    the only way ESPN exposes player-level data. Safe to rerun; always
    refetches from scratch. Also rebuilds player_availability.csv (played/DNP
    per rostered athlete) from the same fetched summaries.
    """
    results = load_results()
    if "event_id" not in results.columns:
        raise RuntimeError("results.csv has no event_id column -- rerun wnba.data.espn_ingest.refresh() first")

    cutoff = results["date"].max() - pd.Timedelta(days=365 * lookback_years)
    games = results[results["date"] >= cutoff].dropna(subset=["event_id"])

    rows = []
    availability_rows = []
    for i, game in enumerate(games.itertuples(index=False)):
        if i > 0:
            time.sleep(PLAYER_BOX_MIN_REQUEST_INTERVAL)
        event_id = str(int(game.event_id))
        summary = _fetch_boxscore(event_id)
        date = game.date.strftime("%Y-%m-%d")
        season, is_playoff = int(game.season), bool(game.is_playoff)
        rows.extend(_parse_boxscore(summary, event_id, date=date, season=season, is_playoff=is_playoff))
        availability_rows.extend(_parse_availability(summary, event_id, date=date, season=season, is_playoff=is_playoff))

    df = pd.DataFrame(rows)
    df.to_csv(_PLAYER_BOX_FILE, index=False)

    availability_df = pd.DataFrame(availability_rows)
    availability_df.to_csv(_PLAYER_AVAILABILITY_FILE, index=False)
    return df


def load_player_boxscores() -> pd.DataFrame:
    if not _PLAYER_BOX_FILE.exists():
        raise FileNotFoundError(f"{_PLAYER_BOX_FILE} not found. Run refresh() first.")
    df = pd.read_csv(_PLAYER_BOX_FILE, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_player_availability() -> pd.DataFrame:
    if not _PLAYER_AVAILABILITY_FILE.exists():
        raise FileNotFoundError(f"{_PLAYER_AVAILABILITY_FILE} not found. Run refresh() first.")
    df = pd.read_csv(_PLAYER_AVAILABILITY_FILE, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)
