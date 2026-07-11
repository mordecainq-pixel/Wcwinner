"""Historical + live WNBA game data from ESPN's public scoreboard API.

No auth, no key, no rate-limit documentation found (being conservative with
spacing anyway). Confirmed live: works for historical dates back to at least
1998, and a `dates=YYYYMMDD-YYYYMMDD` range with an explicit `limit` pulls a
whole season in one request (the default page size silently truncates to
100, well under a real season's ~300+ games including playoffs).
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests

from wnba.config import DATA_RAW_DIR, ESPN_BASE_URL, ESPN_HISTORICAL_LOOKBACK_YEARS, ESPN_REQUEST_LIMIT

_RESULTS_FILE = DATA_RAW_DIR / "results.csv"
_MIN_REQUEST_INTERVAL = 1.0  # seconds; conservative self-throttle, no documented limit to target


def _fetch_season_events(year: int) -> list[dict]:
    resp = requests.get(
        f"{ESPN_BASE_URL}/scoreboard",
        params={"dates": f"{year}0401-{year}1130", "limit": ESPN_REQUEST_LIMIT},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("events", [])


def _parse_event(event: dict) -> dict | None:
    comp = event["competitions"][0]
    if not comp["status"]["type"]["completed"]:
        return None  # future/in-progress fixture, not a played game
    if comp.get("type", {}).get("abbreviation") == "ALLSTAR":
        return None  # exhibition game with made-up "Team Collier vs Team Clark"
        # style rosters, not real franchises -- would corrupt team ratings
    if event["season"]["type"] not in (2, 3):
        return None  # season.type 1 = preseason: includes exhibition games
        # against non-WNBA national teams (Japan, Australia, Puerto Rico...)
        # ahead of Olympics -- real opponents but not real competitive games

    competitors = {c["homeAway"]: c for c in comp["competitors"]}
    if "home" not in competitors or "away" not in competitors:
        return None

    home, away = competitors["home"], competitors["away"]
    return {
        "date": event["date"][:10],
        "home_team": home["team"]["displayName"],
        "away_team": away["team"]["displayName"],
        "home_score": int(home["score"]),
        "away_score": int(away["score"]),
        "is_playoff": event["season"]["type"] == 3,
        "neutral": bool(comp.get("neutralSite", False)),
        "season": event["season"]["year"],
    }


def refresh(lookback_years: int = ESPN_HISTORICAL_LOOKBACK_YEARS) -> pd.DataFrame:
    """Pulls every completed game from the last `lookback_years` WNBA seasons
    (one request per season) and writes them to data/wnba_raw/results.csv.
    Safe to rerun anytime -- always refetches, no incremental/cache logic,
    since it's cheap (one request per season).
    """
    current_year = date.today().year
    rows = []
    for i, year in enumerate(range(current_year - lookback_years + 1, current_year + 1)):
        if i > 0:
            time.sleep(_MIN_REQUEST_INTERVAL)
        events = _fetch_season_events(year)
        for event in events:
            parsed = _parse_event(event)
            if parsed:
                rows.append(parsed)

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "home_team", "away_team"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(_RESULTS_FILE, index=False)
    return df


def load_results() -> pd.DataFrame:
    if not _RESULTS_FILE.exists():
        raise FileNotFoundError(f"{_RESULTS_FILE} not found. Run refresh() first.")
    df = pd.read_csv(_RESULTS_FILE, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def data_freshness() -> dict:
    df = load_results()
    return {
        "rows_total": len(df),
        "max_date": df["date"].max(),
        "min_date": df["date"].min(),
        "seasons_covered": sorted(df["season"].unique().tolist()),
    }
