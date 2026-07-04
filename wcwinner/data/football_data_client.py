"""football-data.org client — the live source of truth for World Cup 2026
fixtures, results, and bracket state.

Confirmed against the live free tier: it pre-seeds the entire knockout bracket
through the Final, with real team names filled in as each round resolves and
None/None placeholders for slots that aren't decided yet. That means bracket
state here is fully API-driven; no manual web-search confirmation is needed
for normal operation.
"""
from __future__ import annotations

import time

import requests

from wcwinner.config import FOOTBALL_DATA_API_TOKEN, FOOTBALL_DATA_WC_CODE
from wcwinner.data.team_names import to_canonical

_BASE_URL = "https://api.football-data.org/v4"
_MIN_REQUEST_INTERVAL = 6.5  # seconds; free tier allows 10 req/min

# Module-level (not per-instance) so the spacing actually holds across calls
# that each construct their own FootballDataClient -- bracket_state() does
# exactly that on every call, so per-instance timing never did anything.
_last_request_ts = 0.0


class FootballDataClient:
    def __init__(self, token: str | None = None):
        self.token = token or FOOTBALL_DATA_API_TOKEN
        if not self.token:
            raise RuntimeError("FOOTBALL_DATA_API_TOKEN is not set in .env")

    def _get(self, path: str, params: dict | None = None) -> dict:
        global _last_request_ts
        elapsed = time.monotonic() - _last_request_ts
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

        for attempt in range(3):
            resp = requests.get(
                f"{_BASE_URL}{path}",
                headers={"X-Auth-Token": self.token},
                params=params,
                timeout=15,
            )
            _last_request_ts = time.monotonic()
            if resp.status_code == 429 and attempt < 2:
                time.sleep(15)  # free tier resets quickly; back off and retry rather than crash
                continue
            resp.raise_for_status()
            return resp.json()

    def competition_info(self, code: str = FOOTBALL_DATA_WC_CODE) -> dict:
        return self._get(f"/competitions/{code}")

    def matches(self, code: str = FOOTBALL_DATA_WC_CODE, status: str | None = None) -> list[dict]:
        params = {"status": status} if status else None
        return self._get(f"/competitions/{code}/matches", params=params)["matches"]

    def standings(self, code: str = FOOTBALL_DATA_WC_CODE) -> list[dict]:
        return self._get(f"/competitions/{code}/standings")["standings"]


def _canonicalize_match(m: dict) -> dict:
    home = m["homeTeam"]["name"]
    away = m["awayTeam"]["name"]
    return {
        "match_id": m["id"],
        "utc_date": m["utcDate"],
        "status": m["status"],
        "stage": m["stage"],
        "group": m.get("group"),
        "home_team": to_canonical(home) if home else None,
        "away_team": to_canonical(away) if away else None,
        "home_score": m["score"]["fullTime"]["home"],
        "away_score": m["score"]["fullTime"]["away"],
        "winner": m["score"].get("winner"),
    }


def bracket_state(client: FootballDataClient | None = None) -> dict:
    """Structured snapshot of the tournament: matchday, per-stage match lists,
    and group tables. This is the API-driven "what round are we in / who plays
    who next" answer the CLI and bracket sim consume.
    """
    client = client or FootballDataClient()
    info = client.competition_info()
    raw_matches = client.matches()
    matches = [_canonicalize_match(m) for m in raw_matches]

    groups = {}
    for group_table in client.standings():
        if group_table.get("group"):
            groups[group_table["group"]] = [
                {
                    "team": to_canonical(row["team"]["name"]),
                    "played": row["playedGames"],
                    "points": row["points"],
                    "goal_difference": row["goalDifference"],
                }
                for row in group_table["table"]
            ]

    return {
        "current_matchday": info["currentSeason"]["currentMatchday"],
        "season_start": info["currentSeason"]["startDate"],
        "season_end": info["currentSeason"]["endDate"],
        "matches": matches,
        "groups": groups,
    }
