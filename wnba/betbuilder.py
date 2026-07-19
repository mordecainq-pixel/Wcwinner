"""Bet-slip builder for WNBA: game legs (h2h/totals/spread, no draw outcome
like soccer has) plus player-prop legs (over/under stat lines and
double-double/triple-double milestones), all in the same combinable pool.

Same two usage modes as wcwinner/betbuilder.py: browse the single best-EV
leg per match, or pick specific matches and search for a target payout
scoped to only those matches (never the whole slate -- see that module's
docstring for why unscoped target search produces forced longshot parlays).

Player-prop legs are only built for (player, stat) pairs the market
actually quotes a line for, not the model's own fair odds -- there's no
useful "bet" if there's no real market to bet it on, unlike game h2h which
practically always has a market this module can fall back to.

This does not recommend real-money betting. It surfaces the model's read
on value -- nothing here is a guarantee, and every rendered output says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from wnba.data.odds_client import OddsAPIClient, game_market_probabilities, player_milestone_odds, player_prop_lines
from wnba.model import margin_model
from wnba.model.player_model import STAT_LABELS, prob_double_double, prob_over, prob_triple_double, simulate_player_games

# Confirmed live against real upcoming games (see data/odds_client.py) once
# both the "us" and "us2" regions are queried together.
KNOWN_BOOKMAKERS = ["fanduel", "draftkings", "espnbet", "betmgm", "betrivers", "hardrockbet", "betparx", "fliff", "betonlineag", "prophetx"]


@dataclass
class Leg:
    home_team: str
    away_team: str
    date: str
    market_type: str  # "h2h", "totals", "spread", "prop", or "milestone"
    pick: str  # h2h/spread: "home"/"away"; totals/prop: "over"/"under"; milestone: "yes"
    line: float | None
    model_prob: float
    odds: float
    market_prob: float | None
    has_market_data: bool
    player_name: str | None = None
    stat: str | None = None

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
            team = self.home_team if self.pick == "home" else self.away_team
            return f"{team} to win"
        if self.market_type == "totals":
            return f"{'Over' if self.pick == 'over' else 'Under'} {self.line} total points"
        if self.market_type == "spread":
            team = self.home_team if self.pick == "home" else self.away_team
            return f"{team} {self.line:+.1f}"
        stat_label = STAT_LABELS.get(self.stat, self.stat)
        if self.market_type == "prop":
            return f"{self.player_name} {'Over' if self.pick == 'over' else 'Under'} {self.line} {stat_label}"
        return f"{self.player_name} {stat_label} (Yes)"

    @property
    def reasoning(self) -> str:
        if self.market_prob is None:
            return "no market data to compare against; priced at the model's own fair odds"
        note = " (market side not de-vigged -- book only quotes 'Yes')" if self.market_type == "milestone" else ""
        return f"model gives this {self.model_prob*100:.0f}% vs. the market's {self.market_prob*100:.0f}%{note} ({self.ev*100:+.0f}% EV)"


def _h2h_candidates(home, away, date, win_prob, market_row) -> list[Leg]:
    legs = []
    for pick, model_prob in (("home", win_prob), ("away", 1.0 - win_prob)):
        if market_row is not None and pd.notna(market_row.get("market_home_prob")):
            odds = market_row["market_home_odds"] if pick == "home" else market_row["market_away_odds"]
            m_prob = market_row["market_home_prob"] if pick == "home" else market_row["market_away_prob"]
            has_market = True
        else:
            odds, m_prob, has_market = 1.0 / model_prob, None, False
        legs.append(Leg(home, away, date, "h2h", pick, None, model_prob, odds, m_prob, has_market))
    return legs


def _totals_candidates(home, away, date, model, market_row) -> list[Leg]:
    if market_row is not None and pd.notna(market_row.get("total_line")):
        line = market_row["total_line"]
    else:
        mu, _ = model.total_distribution(home, away, neutral=False)
        line = round(mu * 2) / 2  # nearest half-point, a reasonable no-market fallback
    p_over = model.prob_over(home, away, line, neutral=False)

    legs = []
    for pick, model_prob in (("over", p_over), ("under", 1.0 - p_over)):
        if market_row is not None and pd.notna(market_row.get("total_over_prob")):
            m_prob = market_row["total_over_prob"] if pick == "over" else market_row["total_under_prob"]
            odds = 1.0 / m_prob if m_prob else 1.0 / model_prob
            has_market = True
        else:
            odds, m_prob, has_market = 1.0 / model_prob, None, False
        legs.append(Leg(home, away, date, "totals", pick, line, model_prob, odds, m_prob, has_market))
    return legs


def _spread_candidates(home, away, date, model, market_row) -> list[Leg]:
    if market_row is None or pd.isna(market_row.get("home_spread_point")):
        return []
    home_point = market_row["home_spread_point"]
    p_home_covers = model.prob_home_covers_spread(home, away, home_point, neutral=False)
    home_prob = market_row["home_spread_prob"]
    return [
        Leg(home, away, date, "spread", "home", home_point, p_home_covers, 1.0 / home_prob, home_prob, True),
        Leg(home, away, date, "spread", "away", -home_point, 1.0 - p_home_covers, 1.0 / (1 - home_prob), 1.0 - home_prob, True),
    ]


def _resolve_player_id(player_box: pd.DataFrame, name: str) -> str | None:
    matches = player_box[player_box["player_name"].str.lower() == name.lower()]
    if matches.empty:
        return None
    return matches.iloc[-1]["player_id"]


def out_player_ids_for_team(injuries: pd.DataFrame, team: str) -> set[str]:
    """Currently-Out player_ids for `team`, from the live injury report.
    Only `status == "Out"` counts -- Questionable/Day-To-Day aren't reliable
    enough to condition a projection on before the game actually happens.
    Rows with no resolved player_id (a link-parsing miss) are dropped rather
    than risk a bad match.
    """
    out = injuries[(injuries["team"] == team) & (injuries["status"] == "Out") & injuries["player_id"].notna()]
    return set(out["player_id"])


def _prop_leg_candidates(home, away, date, opponent_for_player, player_box, opponent_factors, prop_df, availability=None, out_ids=None) -> list[Leg]:
    legs = []
    for row in prop_df.itertuples(index=False):
        player_id = _resolve_player_id(player_box, row.player_name)
        if player_id is None:
            continue
        try:
            sim = simulate_player_games(
                player_box, player_id, opponent_for_player, opponent_factors,
                out_player_ids=out_ids, availability=availability,
            )
        except ValueError:
            continue  # not enough game history for this player yet
        p_over = prob_over(sim, row.stat, row.line)
        for pick, model_prob, odds, m_prob in (
            ("over", p_over, row.over_odds, row.market_over_prob),
            ("under", 1.0 - p_over, row.under_odds, row.market_under_prob),
        ):
            legs.append(Leg(
                home, away, date, "prop", pick, row.line, model_prob, odds, m_prob, True,
                player_name=row.player_name, stat=row.stat,
            ))
    return legs


def _milestone_leg_candidates(home, away, date, opponent_for_player, player_box, opponent_factors, milestone_df, availability=None, out_ids=None) -> list[Leg]:
    legs = []
    for row in milestone_df.itertuples(index=False):
        player_id = _resolve_player_id(player_box, row.player_name)
        if player_id is None:
            continue
        try:
            sim = simulate_player_games(
                player_box, player_id, opponent_for_player, opponent_factors,
                out_player_ids=out_ids, availability=availability,
            )
        except ValueError:
            continue
        model_prob = prob_double_double(sim) if row.stat == "double_double" else prob_triple_double(sim)
        legs.append(Leg(
            home, away, date, "milestone", "yes", None, model_prob, row.yes_odds, row.market_yes_prob_raw, True,
            player_name=row.player_name, stat=row.stat,
        ))
    return legs


@dataclass
class MatchOptions:
    home_team: str
    away_team: str
    date: str
    legs: list[Leg]
    event_id: str | None = None  # Odds-API event id, needed to fetch props for this match specifically

    @property
    def best_leg(self) -> Leg:
        return max(self.legs, key=lambda l: (l.ev, l.model_prob))


def gather_match_options(bookmakers: str | None = None) -> list[MatchOptions]:
    """Every upcoming game with its GAME-LINE bet candidates only (h2h,
    totals, spread) -- cheap, one bulk market request for the whole slate.

    Player props are deliberately NOT fetched here. They're priced per
    market x region on The Odds API and get expensive fast across a whole
    slate (see config.py's ODDS_API_REGIONS comment for the real numbers
    from testing this) -- call `add_player_prop_legs` for specific games
    the user has actually picked instead.
    """
    model = margin_model.load()
    odds_client = OddsAPIClient()

    try:
        game_odds_df = game_market_probabilities(client=odds_client, bookmakers=bookmakers)
        game_odds_by_fixture = {(r.home_team, r.away_team): r._asdict() for r in game_odds_df.itertuples(index=False)}
    except Exception:
        game_odds_by_fixture = {}

    try:
        events = odds_client.events()
    except Exception:
        events = []

    results = []
    for event in events:
        home, away, date = event["home_team"], event["away_team"], event["commence_time"][:10]
        win_prob = model.win_probability(home, away, neutral=False)
        market_row = game_odds_by_fixture.get((home, away))

        legs = (
            _h2h_candidates(home, away, date, win_prob, market_row)
            + _totals_candidates(home, away, date, model, market_row)
            + _spread_candidates(home, away, date, model, market_row)
        )
        results.append(MatchOptions(home, away, date, legs, event_id=event["id"]))

    return results


def add_player_prop_legs(
    match: MatchOptions,
    player_box: pd.DataFrame,
    opponent_factors: pd.DataFrame,
    bookmakers: str | None = None,
    client: OddsAPIClient | None = None,
    availability: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
) -> MatchOptions:
    """Fetches and appends player-prop + milestone legs for ONE specific
    match (needs `match.event_id`, set by `gather_match_options`). This is
    the expensive call -- only invoke it for games the user has actually
    selected, not for a whole slate up front.

    `availability` + `injuries` are both optional and default to no
    adjustment -- pass both to condition each side's projections on the
    OTHER team's currently-Out rotation players (see
    model/player_model.py's compute_out_player_adjustment).
    """
    if match.event_id is None:
        return match
    client = client or OddsAPIClient()
    home, away, date = match.home_team, match.away_team, match.date

    try:
        prop_df = player_prop_lines(match.event_id, client=client, bookmakers=bookmakers)
        milestone_df = player_milestone_odds(match.event_id, client=client, bookmakers=bookmakers)
    except Exception:
        return match

    new_legs = list(match.legs)
    for team, opponent in ((home, away), (away, home)):
        team_players = player_box[player_box["team"] == team]["player_name"].unique()
        out_ids = out_player_ids_for_team(injuries, opponent) if injuries is not None else None
        if not prop_df.empty:
            team_prop_df = prop_df[prop_df["player_name"].isin(team_players)]
            new_legs += _prop_leg_candidates(
                home, away, date, opponent, player_box, opponent_factors, team_prop_df,
                availability=availability, out_ids=out_ids,
            )
        if not milestone_df.empty:
            team_milestone_df = milestone_df[milestone_df["player_name"].isin(team_players)]
            new_legs += _milestone_leg_candidates(
                home, away, date, opponent, player_box, opponent_factors, team_milestone_df,
                availability=availability, out_ids=out_ids,
            )

    return MatchOptions(home, away, date, new_legs, event_id=match.event_id)


def gather_candidate_legs(bookmakers: str | None = None) -> list[Leg]:
    """One Leg per match (game lines only): the single highest-EV pick,
    ranked best-to-worst.
    """
    matches = gather_match_options(bookmakers)
    return sorted((m.best_leg for m in matches if m.legs), key=lambda l: l.ev, reverse=True)


def find_best_combo(chosen_matches: list[MatchOptions], target_payout: float) -> list[Leg]:
    """Same target-payout search as wcwinner's, scoped to `chosen_matches`
    only, positive-EV-only where possible -- see this module's docstring.
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
    if not legs:
        return None
    combined_odds = 1.0
    combined_prob = 1.0
    for leg in legs:
        combined_odds *= leg.odds
        combined_prob *= leg.model_prob
    return ParlayResult(list(legs), combined_odds, combined_prob, stake)
