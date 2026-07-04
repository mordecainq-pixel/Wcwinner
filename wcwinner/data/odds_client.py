"""The Odds API client — used strictly for post-hoc validation/calibration
(comparing our model's implied probabilities against the market), never as a
model input.
"""
from __future__ import annotations

import pandas as pd
import requests

from wcwinner.config import ODDS_API_KEY, ODDS_API_WC_SPORT_KEY
from wcwinner.data.team_names import to_canonical

_BASE_URL = "https://api.the-odds-api.com/v4"


class OddsAPIClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or ODDS_API_KEY
        if not self.api_key:
            raise RuntimeError("ODDS_API_KEY is not set in .env")
        self.remaining_credits: int | None = None

    def _get(self, path: str, params: dict) -> list:
        resp = requests.get(
            f"{_BASE_URL}{path}",
            params={**params, "apiKey": self.api_key},
            timeout=15,
        )
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            self.remaining_credits = int(remaining)
        return resp.json()

    def wc_odds(self, regions: str = "us,uk,eu", markets: str = "h2h") -> list[dict]:
        return self._get(
            f"/sports/{ODDS_API_WC_SPORT_KEY}/odds/",
            {"regions": regions, "markets": markets},
        )


def _decimal_to_implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def market_probabilities(client: OddsAPIClient | None = None) -> pd.DataFrame:
    """One row per upcoming WC fixture with the market's de-vigged implied
    win/draw/loss probabilities, averaged across bookmakers.
    """
    client = client or OddsAPIClient()
    events = client.wc_odds()

    rows = []
    for event in events:
        home = to_canonical(event["home_team"])
        away = to_canonical(event["away_team"])
        home_prices, draw_prices, away_prices = [], [], []
        for bookmaker in event["bookmakers"]:
            for market in bookmaker["markets"]:
                if market["key"] != "h2h":
                    continue
                for outcome in market["outcomes"]:
                    if outcome["name"] == event["home_team"]:
                        home_prices.append(outcome["price"])
                    elif outcome["name"] == event["away_team"]:
                        away_prices.append(outcome["price"])
                    elif outcome["name"] == "Draw":
                        draw_prices.append(outcome["price"])

        if not (home_prices and away_prices and draw_prices):
            continue

        avg_home_odds = sum(home_prices) / len(home_prices)
        avg_draw_odds = sum(draw_prices) / len(draw_prices)
        avg_away_odds = sum(away_prices) / len(away_prices)

        raw_home = _decimal_to_implied_prob(avg_home_odds)
        raw_draw = _decimal_to_implied_prob(avg_draw_odds)
        raw_away = _decimal_to_implied_prob(avg_away_odds)
        overround = raw_home + raw_draw + raw_away  # >1 due to bookmaker margin

        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "commence_time": event["commence_time"],
                "n_bookmakers": len(event["bookmakers"]),
                "market_home_prob": raw_home / overround,
                "market_draw_prob": raw_draw / overround,
                "market_away_prob": raw_away / overround,
                "overround": overround,
                # Actual average bookmaker decimal prices (vig included) - what you'd
                # really be paid, as opposed to the de-vigged "fair" probabilities above.
                "market_home_odds": avg_home_odds,
                "market_draw_odds": avg_draw_odds,
                "market_away_odds": avg_away_odds,
            }
        )

    return pd.DataFrame(rows)
