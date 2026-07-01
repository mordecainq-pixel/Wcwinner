"""API-Football client — secondary/optional cross-check source.

Confirmed directly against the live free tier that it CANNOT serve the 2026
season ("Free plans do not have access to this season, try from 2022 to
2024") and also blocks the `next` fixtures parameter. It therefore plays no
role in live 2026 data; its only remaining use is spot-checking the Kaggle
dataset's coverage of 2022-2024 international matches, which is a nice-to-have,
not a dependency.

Supports both the direct api-sports.io dashboard auth (`x-apisports-key`
header) and a RapidAPI-mirrored account, since the direct host is blocked by
some sandboxed network policies while the RapidAPI mirror is not. Whichever
credential is present in .env is used; direct api-sports.io takes priority.
"""
from __future__ import annotations

import requests

from wcwinner.config import (
    API_FOOTBALL_HOST,
    API_FOOTBALL_KEY,
    API_FOOTBALL_WC_LEAGUE_ID,
    RAPIDAPI_FOOTBALL_HOST,
    RAPIDAPI_KEY,
)

FREE_TIER_SEASONS = (2022, 2023, 2024)


class APIFootballClient:
    def __init__(self):
        if API_FOOTBALL_KEY:
            self.base_url = f"https://{API_FOOTBALL_HOST}"
            self.headers = {"x-apisports-key": API_FOOTBALL_KEY}
        elif RAPIDAPI_KEY:
            self.base_url = f"https://{RAPIDAPI_FOOTBALL_HOST}"
            self.headers = {
                "x-rapidapi-key": RAPIDAPI_KEY,
                "x-rapidapi-host": RAPIDAPI_FOOTBALL_HOST,
            }
        else:
            raise RuntimeError(
                "Neither API_FOOTBALL_KEY nor RAPIDAPI_KEY is set in .env"
            )

    def _get(self, path: str, params: dict) -> dict:
        resp = requests.get(f"{self.base_url}{path}", headers=self.headers, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("errors"):
            raise RuntimeError(f"API-Football error for {path} {params}: {payload['errors']}")
        return payload

    def fixtures(self, league: int = API_FOOTBALL_WC_LEAGUE_ID, season: int = 2022) -> list[dict]:
        if season not in FREE_TIER_SEASONS:
            raise ValueError(
                f"Free tier only serves seasons {FREE_TIER_SEASONS}; {season} will 404 with a plan error."
            )
        return self._get("/fixtures", {"league": league, "season": season})["response"]


def coverage_check(kaggle_results, season: int = 2022) -> dict:
    """Compare match counts between the Kaggle dataset and API-Football for a
    World Cup season both can actually serve, to spot missing rows.
    """
    client = APIFootballClient()
    api_fixtures = client.fixtures(season=season)
    api_count = len(api_fixtures)

    kaggle_wc = kaggle_results[
        (kaggle_results["tournament"] == "FIFA World Cup")
        & (kaggle_results["date"].dt.year == season)
    ]
    return {
        "season": season,
        "kaggle_match_count": len(kaggle_wc),
        "api_football_match_count": api_count,
        "difference": api_count - len(kaggle_wc),
    }
