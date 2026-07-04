"""Bet-slip builder.

Two ways to use this, both starting from the same match list:

1. Browse mode (`gather_candidate_legs`): one best-EV pick per match, ranked
   best-to-worst. You pick however many you want; `combine_legs` does the
   payout math for exactly that selection. No search, no target.

2. Target mode (`find_best_combo`): you pick which MATCHES interest you
   (e.g. "France's game and Canada's game"), give a target payout, and it
   searches across every bet type (h2h/totals/spread) for *those matches
   specifically* -- not the whole schedule -- for the one-leg-per-match
   combination closest to your target, tie-broken by the highest combined
   model probability. Scoping the search to matches you've already chosen
   is what makes this safe: an earlier version searched the *whole* upcoming
   schedule for a target and would force together whatever was available
   when the pool was small (e.g. a 3x request landing on a forced 29x
   longshot parlay because only 2 matches existed in a narrow date filter).
   Restricting to a handful of matches you picked, each with ~7 real bet
   options, gives the search room to actually get close -- and if it still
   can't, `find_best_combo` return value makes that easy to detect and
   report honestly (see the `>15% off target` check in the CLI/app).

This does not recommend real-money betting. It surfaces the model's read
on value -- nothing here is a guarantee, and every rendered output says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from wcwinner.data.football_data_client import bracket_state
from wcwinner.data.odds_client import multi_market_odds
from wcwinner.model.dixon_coles import DixonColesModel
from wcwinner.simulate.match import predict_match, prob_home_covers_spread, prob_total_over

DEFAULT_TOTALS_LINE = 2.5

# A sample of sportsbooks The Odds API actually covers for this sport (confirmed
# live). Kalshi and Polymarket are NOT available through this provider.
KNOWN_BOOKMAKERS = ["fanduel", "draftkings", "betmgm", "pinnacle", "williamhill", "bovada", "unibet"]


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

    @property
    def reasoning(self) -> str:
        if self.market_prob is None:
            return "no market data to compare against; priced at the model's own fair odds"
        return f"model gives this {self.model_prob*100:.0f}% vs. the market's {self.market_prob*100:.0f}% ({self.ev*100:+.0f}% EV)"


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


@dataclass
class MatchOptions:
    home_team: str
    away_team: str
    date: str
    legs: list[Leg]  # every bet-type candidate for this match

    @property
    def best_leg(self) -> Leg:
        return max(self.legs, key=lambda l: (l.ev, l.model_prob))


def gather_match_options(
    model: DixonColesModel,
    elo_ratings: dict[str, float],
    bookmakers: str | None = None,
) -> list[MatchOptions]:
    """Every upcoming (not-yet-played) WC 2026 match with its full set of bet
    candidates across h2h/totals/spread. `bookmakers` (e.g. "fanduel" or
    "fanduel,draftkings") restricts odds to those specific sportsbooks
    instead of averaging across every book The Odds API covers -- note this
    trades off market coverage; some books don't quote totals/spreads for
    every match, so a narrower selection means more legs fall back to the
    model's own fair odds with no market to compare against.
    """
    bs = bracket_state()
    upcoming = [
        m for m in bs["matches"]
        if m["status"] in ("TIMED", "SCHEDULED") and m["home_team"] and m["away_team"]
    ]

    try:
        market_by_fixture = {(e["home_team"], e["away_team"]): e for e in multi_market_odds(bookmakers=bookmakers)}
    except Exception:
        market_by_fixture = {}

    results = []
    for m in upcoming:
        home, away, date = m["home_team"], m["away_team"], m["utc_date"][:10]
        pred = predict_match(model, home, away, neutral=True, elo_ratings=elo_ratings)
        model_probs = {"home": pred["p_home_win"], "draw": pred["p_draw"], "away": pred["p_away_win"]}
        market = market_by_fixture.get((home, away), {})

        legs = (
            _h2h_candidates(home, away, date, model_probs, market.get("h2h"))
            + _totals_candidates(home, away, date, pred["score_matrix"], market.get("totals"))
            + _spread_candidates(home, away, date, pred["score_matrix"], market.get("spreads"))
        )
        results.append(MatchOptions(home, away, date, legs))

    return results


def gather_candidate_legs(
    model: DixonColesModel,
    elo_ratings: dict[str, float],
    bookmakers: str | None = None,
) -> list[Leg]:
    """One Leg per match: the single highest-EV pick, ranked best-to-worst.
    For browsing/picking individual legs yourself (see `combine_legs`).
    """
    matches = gather_match_options(model, elo_ratings, bookmakers)
    return sorted((m.best_leg for m in matches), key=lambda l: l.ev, reverse=True)


def find_best_combo(chosen_matches: list[MatchOptions], target_payout: float) -> list[Leg]:
    """Searches every bet-type combination (one leg per chosen match) for
    whichever gets closest to `target_payout`, tie-broken by the highest
    combined model probability. Scoped to `chosen_matches` only -- the
    matches the user picked, not the whole schedule -- so the search space
    is small (at most ~7 options per match) and the result is either a good
    fit or an honestly-reportable miss, never a forced non-sequitur.

    Never knowingly includes a negative-EV leg just to fit the target more
    precisely: each match's option pool is restricted to legs with positive
    EV (or no market data to judge EV at all) before searching, only falling
    back to that match's full option list if every single option reads as
    negative EV (rare, but a match needs *some* leg to be included at all).
    """
    pools = []
    for m in chosen_matches:
        positive_ev = [l for l in m.legs if l.market_prob is None or l.ev > 0]
        pools.append(positive_ev if positive_ev else m.legs)

    best: list[Leg] | None = None
    best_score = None
    for combo in product(*pools):
        combined_odds = 1.0
        combined_prob = 1.0
        for leg in combo:
            combined_odds *= leg.odds
            combined_prob *= leg.model_prob
        score = (abs(combined_odds - target_payout), -combined_prob)
        if best_score is None or score < best_score:
            best_score = score
            best = list(combo)
    return best


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
    """Payout/probability math for exactly the legs given."""
    if not legs:
        return None
    combined_odds = 1.0
    combined_prob = 1.0
    for leg in legs:
        combined_odds *= leg.odds
        combined_prob *= leg.model_prob
    return ParlayResult(list(legs), combined_odds, combined_prob, stake)
