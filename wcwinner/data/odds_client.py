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

    def wc_odds_multi_market(self, regions: str = "us,uk,eu") -> list[dict]:
        """h2h + totals (over/under) + spreads (handicap). BTTS isn't offered
        by this provider for soccer ("Markets not supported by this endpoint:
        btts" - confirmed directly against the live API)."""
        return self._get(
            f"/sports/{ODDS_API_WC_SPORT_KEY}/odds/",
            {"regions": regions, "markets": "h2h,totals,spreads"},
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


def _average_by_most_common_point(entries: list[tuple[float, float]]) -> tuple[float, float] | None:
    """entries: list of (point, price). Averages prices for whichever point
    value is most common across bookmakers, ignoring bookmakers quoting a
    different line, so a 2.5 line doesn't get blended with a stray 3.5 line.
    """
    if not entries:
        return None
    from collections import Counter

    point_counts = Counter(p for p, _ in entries)
    common_point = point_counts.most_common(1)[0][0]
    prices = [price for p, price in entries if p == common_point]
    return common_point, sum(prices) / len(prices)


def multi_market_odds(client: OddsAPIClient | None = None) -> list[dict]:
    """One entry per upcoming WC fixture with h2h, totals, and spreads market
    data (whichever are available), used by the bet builder to consider more
    than just the match-result market.
    """
    client = client or OddsAPIClient()
    events = client.wc_odds_multi_market()

    results = []
    for event in events:
        home_raw, away_raw = event["home_team"], event["away_team"]
        home, away = to_canonical(home_raw), to_canonical(away_raw)

        h2h_home, h2h_draw, h2h_away = [], [], []
        totals_over, totals_under = [], []
        spreads_home, spreads_away = [], []

        for bookmaker in event["bookmakers"]:
            for market in bookmaker["markets"]:
                if market["key"] == "h2h":
                    for o in market["outcomes"]:
                        if o["name"] == home_raw:
                            h2h_home.append(o["price"])
                        elif o["name"] == away_raw:
                            h2h_away.append(o["price"])
                        elif o["name"] == "Draw":
                            h2h_draw.append(o["price"])
                elif market["key"] == "totals":
                    for o in market["outcomes"]:
                        if o["name"] == "Over":
                            totals_over.append((o["point"], o["price"]))
                        elif o["name"] == "Under":
                            totals_under.append((o["point"], o["price"]))
                elif market["key"] == "spreads":
                    for o in market["outcomes"]:
                        if o["name"] == home_raw:
                            spreads_home.append((o["point"], o["price"]))
                        elif o["name"] == away_raw:
                            spreads_away.append((o["point"], o["price"]))

        entry = {"home_team": home, "away_team": away, "commence_time": event["commence_time"], "h2h": None, "totals": None, "spreads": None}

        if h2h_home and h2h_draw and h2h_away:
            avg_h, avg_d, avg_a = sum(h2h_home) / len(h2h_home), sum(h2h_draw) / len(h2h_draw), sum(h2h_away) / len(h2h_away)
            overround = _decimal_to_implied_prob(avg_h) + _decimal_to_implied_prob(avg_d) + _decimal_to_implied_prob(avg_a)
            entry["h2h"] = {
                "home_odds": avg_h, "draw_odds": avg_d, "away_odds": avg_a,
                "home_prob": _decimal_to_implied_prob(avg_h) / overround,
                "draw_prob": _decimal_to_implied_prob(avg_d) / overround,
                "away_prob": _decimal_to_implied_prob(avg_a) / overround,
            }

        over = _average_by_most_common_point(totals_over)
        under = _average_by_most_common_point(totals_under)
        if over and under and over[0] == under[0]:
            overround = _decimal_to_implied_prob(over[1]) + _decimal_to_implied_prob(under[1])
            entry["totals"] = {
                "line": over[0], "over_odds": over[1], "under_odds": under[1],
                "over_prob": _decimal_to_implied_prob(over[1]) / overround,
                "under_prob": _decimal_to_implied_prob(under[1]) / overround,
            }

        home_spread = _average_by_most_common_point(spreads_home)
        away_spread = _average_by_most_common_point(spreads_away)
        if home_spread and away_spread:
            overround = _decimal_to_implied_prob(home_spread[1]) + _decimal_to_implied_prob(away_spread[1])
            entry["spreads"] = {
                "home_point": home_spread[0], "away_point": away_spread[0],
                "home_odds": home_spread[1], "away_odds": away_spread[1],
                "home_prob": _decimal_to_implied_prob(home_spread[1]) / overround,
                "away_prob": _decimal_to_implied_prob(away_spread[1]) / overround,
            }

        results.append(entry)

    return results
