"""Parlay / bet-slip builder.

Give it a target payout multiple (e.g. 5x), a date range, and optionally how
many legs you want, and it searches upcoming World Cup fixtures across three
market types -- match result (h2h), total goals (over/under), and handicap
(spread) -- for the combination that reaches that payout using real market
odds, prioritizing legs with positive expected value: the model's probability
of that outcome times the real payout, minus 1. That's the actual "does the
model think this is a good bet" test, not just "does the model's probability
exceed the market's" -- a small probability edge can still be a bad bet once
the bookmaker's margin is priced in, and EV is what accounts for that.

Only one leg is taken per match (the single highest-EV option across all
three market types for that match), which keeps legs statistically
independent of each other -- combining two bets from the *same* match (e.g.
"Home wins" and "Over 2.5") would violate the independence assumption the
combined-odds/combined-probability math relies on, since those two outcomes
are correlated within one game.

This does not recommend real-money betting. It surfaces what the model's
edge, if real, would imply about a same-day multi -- nothing here is a
guarantee, and every rendered output says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from wcwinner.data.football_data_client import bracket_state
from wcwinner.data.odds_client import multi_market_odds
from wcwinner.model.dixon_coles import DixonColesModel
from wcwinner.simulate.match import predict_match, prob_home_covers_spread, prob_total_over

DEFAULT_TOTALS_LINE = 2.5


@dataclass
class Leg:
    home_team: str
    away_team: str
    date: str
    market_type: str  # "h2h", "totals", or "spread"
    pick: str  # h2h: "home"/"draw"/"away"; totals: "over"/"under"; spread: "home"/"away"
    line: float | None  # goal line (totals) or handicap point (spread); None for h2h
    model_prob: float
    odds: float  # decimal odds actually used for payout math
    market_prob: float | None  # None if no market coverage for this leg
    has_market_data: bool

    @property
    def edge(self) -> float | None:
        return None if self.market_prob is None else self.model_prob - self.market_prob

    @property
    def ev(self) -> float:
        """Expected value per dollar staked: model_prob * real_payout_odds - 1.
        This is the actual "does the model think this is a good bet" test --
        unlike `edge`, it accounts for the bookmaker's margin, not just
        whether the model's probability beats the de-vigged market number.
        """
        return self.model_prob * self.odds - 1.0

    @property
    def pick_label(self) -> str:
        if self.market_type == "h2h":
            return {"home": f"{self.home_team} to win", "draw": "Draw", "away": f"{self.away_team} to win"}[self.pick]
        if self.market_type == "totals":
            return f"{'Over' if self.pick == 'over' else 'Under'} {self.line} goals"
        team = self.home_team if self.pick == "home" else self.away_team
        return f"{team} {self.line:+.1f}"


def _h2h_candidates(home, away, date, model_probs, market_h2h) -> list[Leg]:
    legs = []
    for pick in ("home", "draw", "away"):
        model_prob = model_probs[pick]
        if market_h2h:
            odds, market_prob, has_market = market_h2h[f"{pick}_odds"], market_h2h[f"{pick}_prob"], True
        else:
            odds, market_prob, has_market = 1.0 / model_prob, None, False
        legs.append(Leg(home, away, date, "h2h", pick, None, model_prob, odds, market_prob, has_market))
    return legs


def _totals_candidates(home, away, date, matrix, market_totals) -> list[Leg]:
    line = market_totals["line"] if market_totals else DEFAULT_TOTALS_LINE
    p_over = prob_total_over(matrix, line)
    legs = []
    for pick, model_prob in (("over", p_over), ("under", 1.0 - p_over)):
        if market_totals:
            odds, market_prob, has_market = market_totals[f"{pick}_odds"], market_totals[f"{pick}_prob"], True
        else:
            odds, market_prob, has_market = 1.0 / model_prob, None, False
        legs.append(Leg(home, away, date, "totals", pick, line, model_prob, odds, market_prob, has_market))
    return legs


def _spread_candidates(home, away, date, matrix, market_spreads) -> list[Leg]:
    if not market_spreads:
        return []  # no sensible default handicap line to offer without the market's guidance
    home_point = market_spreads["home_point"]
    p_home_covers = prob_home_covers_spread(matrix, home_point)
    return [
        Leg(home, away, date, "spread", "home", home_point, p_home_covers, market_spreads["home_odds"], market_spreads["home_prob"], True),
        Leg(home, away, date, "spread", "away", market_spreads["away_point"], 1.0 - p_home_covers, market_spreads["away_odds"], market_spreads["away_prob"], True),
    ]


def gather_candidate_legs(
    model: DixonColesModel,
    elo_ratings: dict[str, float],
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[Leg]:
    """One Leg per not-yet-played WC 2026 match in the date range: the single
    highest-EV pick across h2h, totals, and spread markets for that match.
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
        market_by_fixture = {(e["home_team"], e["away_team"]): e for e in multi_market_odds()}
    except Exception:
        market_by_fixture = {}

    best_legs = []
    for m in upcoming:
        home, away, date = m["home_team"], m["away_team"], m["utc_date"][:10]
        pred = predict_match(model, home, away, neutral=True, elo_ratings=elo_ratings)
        model_probs = {"home": pred["p_home_win"], "draw": pred["p_draw"], "away": pred["p_away_win"]}
        market = market_by_fixture.get((home, away), {})

        candidates = (
            _h2h_candidates(home, away, date, model_probs, market.get("h2h"))
            + _totals_candidates(home, away, date, pred["score_matrix"], market.get("totals"))
            + _spread_candidates(home, away, date, pred["score_matrix"], market.get("spreads"))
        )
        best_legs.append(max(candidates, key=lambda l: (l.ev, l.model_prob)))

    return best_legs


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
    highest combined model probability. Restricted to positive-EV legs (or
    legs with no market data to assess) when there are enough of them; falls
    back to the full pool (flagged via `used_non_edge_legs`) only if there
    aren't.
    """
    if not legs:
        return None

    positive_ev_pool = [l for l in legs if l.market_prob is None or l.ev > 0]
    min_needed = n_legs or 2
    pool, used_non_edge_legs = (
        (positive_ev_pool, False)
        if len(positive_ev_pool) >= min_needed
        else (legs, len(positive_ev_pool) < len(legs))
    )

    ranked = sorted(pool, key=lambda l: l.ev, reverse=True)
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
