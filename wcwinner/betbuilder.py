"""Parlay / bet-slip builder.

Give it a target payout multiple (e.g. 5x), a date range, and optionally how
many legs you want, and it searches upcoming World Cup fixtures for the
combination that reaches that multiple using real market odds -- prioritizing
legs where the model's probability exceeds the market's implied probability
(the same "edge" concept as validate/market_compare.py), so "best" means
"most defensible given where we think the market is wrong," not just
"any combination that multiplies to the target number."

This does not recommend real-money betting. It surfaces what the model's
edge, if real, would imply about a same-day multi -- nothing here is a
guarantee, and every rendered output says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from wcwinner.data.football_data_client import bracket_state
from wcwinner.data.odds_client import market_probabilities
from wcwinner.model.dixon_coles import DixonColesModel
from wcwinner.simulate.match import predict_match

_OUTCOME_LABELS = {"home": "{home} to win", "draw": "Draw", "away": "{away} to win"}


@dataclass
class Leg:
    home_team: str
    away_team: str
    date: str
    pick: str  # "home", "draw", or "away"
    model_prob: float
    odds: float  # decimal odds actually used for payout math
    market_prob: float | None  # None if no market coverage for this fixture
    has_market_data: bool

    @property
    def edge(self) -> float | None:
        return None if self.market_prob is None else self.model_prob - self.market_prob

    @property
    def pick_label(self) -> str:
        return _OUTCOME_LABELS[self.pick].format(home=self.home_team, away=self.away_team)


def gather_candidate_legs(
    model: DixonColesModel,
    elo_ratings: dict[str, float],
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[Leg]:
    """One Leg per not-yet-played WC 2026 match in the date range, using
    that match's most probable outcome per the model as the pick.
    """
    bs = bracket_state()
    upcoming = [
        m for m in bs["matches"]
        if m["status"] in ("TIMED", "SCHEDULED") and m["home_team"] and m["away_team"]
    ]
    if date_from:
        upcoming = [m for m in upcoming if m["utc_date"][:10] >= date_from]
    if date_to:
        upcoming = [m for m in upcoming if m["utc_date"][:10] <= date_to]

    try:
        market_df = market_probabilities()
    except Exception:
        market_df = pd.DataFrame()

    legs = []
    for m in upcoming:
        home, away = m["home_team"], m["away_team"]
        pred = predict_match(model, home, away, neutral=True, elo_ratings=elo_ratings)
        probs = {"home": pred["p_home_win"], "draw": pred["p_draw"], "away": pred["p_away_win"]}
        pick = max(probs, key=probs.get)
        model_prob = probs[pick]

        market_row = None
        if not market_df.empty:
            match = market_df[(market_df["home_team"] == home) & (market_df["away_team"] == away)]
            if not match.empty:
                market_row = match.iloc[0]

        if market_row is not None:
            market_prob = float(market_row[f"market_{pick}_prob"])
            odds = float(market_row[f"market_{pick}_odds"])
            has_market_data = True
        else:
            market_prob, odds, has_market_data = None, 1.0 / model_prob, False

        legs.append(Leg(home, away, m["utc_date"][:10], pick, model_prob, odds, market_prob, has_market_data))

    return legs


@dataclass
class ParlayResult:
    legs: list[Leg]
    combined_odds: float
    combined_model_prob: float
    target_payout: float
    stake: float
    used_non_edge_legs: bool

    @property
    def payout(self) -> float:
        return self.stake * self.combined_odds

    @property
    def profit(self) -> float:
        return self.payout - self.stake


def build_parlay(
    legs: list[Leg],
    target_payout: float,
    stake: float = 10.0,
    n_legs: int | None = None,
    max_legs: int = 8,
) -> ParlayResult | None:
    """Exhaustively search combinations of legs for whichever gets closest to
    `target_payout` using real market odds where available, tie-broken by
    highest combined model probability (the combo our model thinks is most
    likely to actually hit). Restricted to legs with positive model-vs-market
    edge when there are enough of them; falls back to the full pool
    (flagged via `used_non_edge_legs`) only if there aren't.
    """
    if not legs:
        return None

    edge_pool = [l for l in legs if l.edge is None or l.edge > 0]
    min_needed = n_legs or 2
    pool, used_non_edge_legs = (edge_pool, False) if len(edge_pool) >= min_needed else (legs, len(edge_pool) < len(legs))

    ranked = sorted(pool, key=lambda l: (l.edge if l.edge is not None else 0.0), reverse=True)
    leg_counts = [n_legs] if n_legs else list(range(2, min(max_legs, len(ranked)) + 1))

    best: ParlayResult | None = None
    best_score = None
    for k in leg_counts:
        if k < 1 or k > len(ranked):
            continue
        for combo in combinations(ranked, k):
            combined_odds = 1.0
            combined_prob = 1.0
            for leg in combo:
                combined_odds *= leg.odds
                combined_prob *= leg.model_prob
            score = (abs(combined_odds - target_payout), -combined_prob)
            if best_score is None or score < best_score:
                best_score = score
                best = ParlayResult(list(combo), combined_odds, combined_prob, target_payout, stake, used_non_edge_legs)

    return best
