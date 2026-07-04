"""Bet-slip builder.

Redesigned from an earlier "search for a combination that hits a target
payout" approach, which broke down when the candidate pool was small (it
would force together whatever legs existed regardless of how far off the
target they landed, rather than admitting it couldn't get close). The fix
is to stop searching/guessing on the user's behalf: `gather_candidate_legs`
surfaces every available match's single best pick (by real expected value
across the h2h/totals/spread markets), ranked best-to-worst, and the caller
(CLI/app) lets the user pick however many of those they actually want.
`combine_legs` just does the payout/probability math for that exact
selection -- no target, no search, no surprises.

This does not recommend real-money betting. It surfaces the model's read
on value -- nothing here is a guarantee, and every rendered output says so.
"""
from __future__ import annotations

from dataclasses import dataclass

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
        """Expected value per dollar staked: model_prob * real_payout_odds - 1."""
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
        return []
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
    """One Leg per not-yet-played WC 2026 match: the single highest-EV pick
    across h2h, totals, and spread markets for that match. `date_from`/
    `date_to` are optional narrowing filters, not required -- leave both
    None to see every available match.
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

    return sorted(best_legs, key=lambda l: l.ev, reverse=True)


@dataclass
class ParlayResult:
    legs: list[Leg]
    combined_odds: float
    combined_model_prob: float
    stake: float

    @property
    def payout(self) -> float:
        return self.stake * self.combined_odds

    @property
    def profit(self) -> float:
        return self.payout - self.stake


def combine_legs(legs: list[Leg], stake: float = 10.0) -> ParlayResult | None:
    """Payout/probability math for exactly the legs given -- no search, no
    target, no substitutions. The user picked these; this just does the math.
    """
    if not legs:
        return None
    combined_odds = 1.0
    combined_prob = 1.0
    for leg in legs:
        combined_odds *= leg.odds
        combined_prob *= leg.model_prob
    return ParlayResult(list(legs), combined_odds, combined_prob, stake)
