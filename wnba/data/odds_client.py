"""The Odds API client for WNBA game lines and player props. Used for
market comparison and Bet Builder legs, never as a model input -- same
boundary as wcwinner/data/odds_client.py.

Confirmed live against real upcoming games: game lines (h2h/spreads/totals)
and most player-prop markets have real bookmaker coverage (FanDuel,
DraftKings, ESPN BET, and others), but only once both the "us" and "us2"
regions are queried together -- "us" alone misses FanDuel/DraftKings player
props on this API, found by testing both, not by assuming one was enough.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
import requests

from wnba.config import ODDS_API_DOUBLE_DOUBLE_MARKET, ODDS_API_KEY, ODDS_API_PLAYER_PROP_MARKETS, ODDS_API_REGIONS, ODDS_API_TRIPLE_DOUBLE_MARKET, ODDS_API_WNBA_SPORT_KEY

_BASE_URL = "https://api.the-odds-api.com/v4"


class OddsAPIClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or ODDS_API_KEY
        if not self.api_key:
            raise RuntimeError("ODDS_API_KEY is not set in .env")
        self.remaining_credits: int | None = None

    def _get(self, path: str, params: dict) -> dict | list:
        resp = requests.get(f"{_BASE_URL}{path}", params={**params, "apiKey": self.api_key}, timeout=15)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            self.remaining_credits = int(remaining)
        return resp.json()

    def events(self) -> list[dict]:
        """Upcoming games with their Odds-API event ids (needed for the
        per-event player-props endpoint, unlike game lines which come back
        in one bulk request).
        """
        return self._get(f"/sports/{ODDS_API_WNBA_SPORT_KEY}/events", {})

    def game_odds(self, regions: str = ODDS_API_REGIONS, bookmakers: str | None = None) -> list[dict]:
        params = {"markets": "h2h,totals,spreads"}
        if bookmakers:
            params["bookmakers"] = bookmakers
        else:
            params["regions"] = regions
        return self._get(f"/sports/{ODDS_API_WNBA_SPORT_KEY}/odds/", params)

    def event_player_props(self, event_id: str, markets: str, regions: str = ODDS_API_REGIONS, bookmakers: str | None = None) -> dict:
        params = {"markets": markets}
        if bookmakers:
            params["bookmakers"] = bookmakers
        else:
            params["regions"] = regions
        return self._get(f"/sports/{ODDS_API_WNBA_SPORT_KEY}/events/{event_id}/odds", params)


def _implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def _average_by_most_common_point(entries: list[tuple[float, float]]) -> tuple[float, float] | None:
    """entries: list of (point, price). Averages prices for whichever line
    value is most common across bookmakers, so a stray outlier line doesn't
    get blended in with the consensus line.
    """
    if not entries:
        return None
    point_counts = Counter(p for p, _ in entries)
    common_point = point_counts.most_common(1)[0][0]
    prices = [price for p, price in entries if p == common_point]
    return common_point, sum(prices) / len(prices)


def game_market_probabilities(client: OddsAPIClient | None = None, bookmakers: str | None = None) -> pd.DataFrame:
    """One row per upcoming game with de-vigged win probabilities plus
    totals/spreads market data where available. No draw outcome, unlike
    soccer.
    """
    client = client or OddsAPIClient()
    events = client.game_odds(bookmakers=bookmakers)

    rows = []
    for event in events:
        home_raw, away_raw = event["home_team"], event["away_team"]
        h2h_home, h2h_away = [], []
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

        row = {
            "home_team": home_raw, "away_team": away_raw, "commence_time": event["commence_time"],
            "n_bookmakers": len(event["bookmakers"]),
            "market_home_prob": None, "market_away_prob": None,
            "market_home_odds": None, "market_away_odds": None,
            "total_line": None, "total_over_prob": None, "total_under_prob": None,
            "home_spread_point": None, "home_spread_prob": None,
        }

        if h2h_home and h2h_away:
            avg_h, avg_a = sum(h2h_home) / len(h2h_home), sum(h2h_away) / len(h2h_away)
            overround = _implied_prob(avg_h) + _implied_prob(avg_a)
            row.update({
                "market_home_odds": avg_h, "market_away_odds": avg_a,
                "market_home_prob": _implied_prob(avg_h) / overround,
                "market_away_prob": _implied_prob(avg_a) / overround,
            })

        over = _average_by_most_common_point(totals_over)
        under = _average_by_most_common_point(totals_under)
        if over and under and over[0] == under[0]:
            overround = _implied_prob(over[1]) + _implied_prob(under[1])
            row.update({
                "total_line": over[0],
                "total_over_prob": _implied_prob(over[1]) / overround,
                "total_under_prob": _implied_prob(under[1]) / overround,
            })

        home_spread = _average_by_most_common_point(spreads_home)
        away_spread = _average_by_most_common_point(spreads_away)
        if home_spread and away_spread:
            overround = _implied_prob(home_spread[1]) + _implied_prob(away_spread[1])
            row.update({
                "home_spread_point": home_spread[0],
                "home_spread_prob": _implied_prob(home_spread[1]) / overround,
            })

        rows.append(row)

    return pd.DataFrame(rows)


def player_prop_lines(
    event_id: str,
    stats: list[str] | None = None,
    client: OddsAPIClient | None = None,
    bookmakers: str | None = None,
) -> pd.DataFrame:
    """Real over/under lines for one game's players, averaged across
    whichever bookmakers offer each line. `stats` restricts to a subset of
    ODDS_API_PLAYER_PROP_MARKETS keys (e.g. ["points", "rebounds"]);
    default is every mapped stat.
    """
    client = client or OddsAPIClient()
    stats = stats or list(ODDS_API_PLAYER_PROP_MARKETS)
    market_keys = [ODDS_API_PLAYER_PROP_MARKETS[s] for s in stats if s in ODDS_API_PLAYER_PROP_MARKETS]
    data = client.event_player_props(event_id, markets=",".join(market_keys), bookmakers=bookmakers)

    stat_by_market = {v: k for k, v in ODDS_API_PLAYER_PROP_MARKETS.items()}
    by_player_stat: dict[tuple[str, str], dict[float, dict[str, list[float]]]] = {}

    for bookmaker in data.get("bookmakers", []):
        for market in bookmaker["markets"]:
            stat = stat_by_market.get(market["key"])
            if stat is None:
                continue
            for o in market["outcomes"]:
                player = o.get("description")
                if player is None:
                    continue
                key = (player, stat)
                by_player_stat.setdefault(key, {}).setdefault(o["point"], {"Over": [], "Under": []})
                by_player_stat[key][o["point"]][o["name"]].append(o["price"])

    rows = []
    for (player, stat), by_point in by_player_stat.items():
        # use whichever line has the most bookmaker agreement (both sides quoted)
        best_point = max(by_point, key=lambda p: len(by_point[p]["Over"]) + len(by_point[p]["Under"]))
        sides = by_point[best_point]
        if not (sides["Over"] and sides["Under"]):
            continue
        over_odds = sum(sides["Over"]) / len(sides["Over"])
        under_odds = sum(sides["Under"]) / len(sides["Under"])
        overround = _implied_prob(over_odds) + _implied_prob(under_odds)
        rows.append({
            "player_name": player, "stat": stat, "line": best_point,
            "over_odds": over_odds, "under_odds": under_odds,
            "market_over_prob": _implied_prob(over_odds) / overround,
            "market_under_prob": _implied_prob(under_odds) / overround,
        })

    return pd.DataFrame(rows)


def player_milestone_odds(event_id: str, client: OddsAPIClient | None = None, bookmakers: str | None = None) -> pd.DataFrame:
    """Double-double / triple-double "Yes" prices. Many books only quote
    the Yes side for these, so there's no "No" price to de-vig against --
    the returned probability is the raw implied probability (book's margin
    still baked in), not a fair/de-vigged one like the over/under markets
    above.
    """
    client = client or OddsAPIClient()
    data = client.event_player_props(
        event_id, markets=f"{ODDS_API_DOUBLE_DOUBLE_MARKET},{ODDS_API_TRIPLE_DOUBLE_MARKET}", bookmakers=bookmakers,
    )
    market_labels = {ODDS_API_DOUBLE_DOUBLE_MARKET: "double_double", ODDS_API_TRIPLE_DOUBLE_MARKET: "triple_double"}

    by_player_stat: dict[tuple[str, str], list[float]] = {}
    for bookmaker in data.get("bookmakers", []):
        for market in bookmaker["markets"]:
            label = market_labels.get(market["key"])
            if label is None:
                continue
            for o in market["outcomes"]:
                if o["name"] != "Yes":
                    continue
                player = o.get("description")
                by_player_stat.setdefault((player, label), []).append(o["price"])

    rows = [
        {"player_name": p, "stat": s, "yes_odds": sum(prices) / len(prices), "market_yes_prob_raw": _implied_prob(sum(prices) / len(prices))}
        for (p, s), prices in by_player_stat.items()
    ]
    return pd.DataFrame(rows)
