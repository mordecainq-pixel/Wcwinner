import pandas as pd

from wnba.betbuilder import Leg, MatchOptions, add_player_prop_legs, combine_legs, find_best_combo


def _leg(home, away, market_type, pick, model_prob, odds, market_prob=None, line=None, player_name=None, stat=None):
    return Leg(home, away, "2026-07-05", market_type, pick, line, model_prob, odds, market_prob, market_prob is not None, player_name, stat)


def test_combine_legs_payout_and_profit():
    legs = [_leg("A", "B", "h2h", "home", 0.5, 2.0, market_prob=0.4), _leg("C", "D", "h2h", "home", 0.5, 2.5, market_prob=0.4)]
    result = combine_legs(legs, stake=20.0)
    assert abs(result.combined_odds - 5.0) < 1e-9
    assert abs(result.payout - 100.0) < 1e-9
    assert abs(result.profit - 80.0) < 1e-9


def test_combine_legs_returns_none_for_empty_list():
    assert combine_legs([]) is None


def test_leg_pick_label_by_market_type():
    h2h_leg = _leg("Aces", "Liberty", "h2h", "home", 0.6, 2.0, market_prob=0.55)
    assert h2h_leg.pick_label == "Aces to win"

    totals_leg = _leg("Aces", "Liberty", "totals", "over", 0.55, 1.9, market_prob=0.5, line=165.5)
    assert totals_leg.pick_label == "Over 165.5 total points"

    spread_leg = _leg("Aces", "Liberty", "spread", "home", 0.6, 1.9, market_prob=0.5, line=-3.5)
    assert spread_leg.pick_label == "Aces -3.5"

    prop_leg = _leg("Aces", "Liberty", "prop", "over", 0.6, 1.9, market_prob=0.5, line=17.5, player_name="Kelsey Mitchell", stat="points")
    assert prop_leg.pick_label == "Kelsey Mitchell Over 17.5 Points"

    milestone_leg = _leg("Aces", "Liberty", "milestone", "yes", 0.4, 1.83, market_prob=0.55, player_name="A'ja Wilson", stat="double_double")
    assert milestone_leg.pick_label == "A'ja Wilson Double-Double (Yes)"


def test_leg_edge_is_none_without_market_data():
    leg = _leg("A", "B", "h2h", "home", 0.5, 2.0, market_prob=None)
    assert leg.edge is None
    assert leg.ev == 0.0  # fair odds (1/model_prob) always give exactly zero EV


def test_leg_no_draw_outcome_exists():
    # WNBA has no draw, unlike soccer's h2h -- only "home"/"away" picks should
    # ever appear for the h2h market type in practice (nothing enforces this
    # at the type level, but the label formatting only handles those two).
    leg = _leg("A", "B", "h2h", "away", 0.4, 2.5, market_prob=0.35)
    assert leg.pick_label == "B to win"


def test_find_best_combo_scoped_to_chosen_matches_only():
    m1 = MatchOptions("A", "B", "2026-07-05", [
        _leg("A", "B", "h2h", "home", 0.6, 1.5, market_prob=0.6),
        _leg("A", "B", "h2h", "away", 0.3, 3.0, market_prob=0.3),
    ])
    m2 = MatchOptions("C", "D", "2026-07-05", [
        _leg("C", "D", "h2h", "home", 0.5, 2.0, market_prob=0.5),
        _leg("C", "D", "h2h", "away", 0.4, 2.4, market_prob=0.4),
    ])
    combo = find_best_combo([m1, m2], target_payout=3.0)
    assert len(combo) == 2
    assert abs(combo[0].odds * combo[1].odds - 3.0) < 1e-9


class _PropsStubClient:
    def __init__(self, prop_response, milestone_response):
        self.prop_response = prop_response
        self.milestone_response = milestone_response

    def event_player_props(self, event_id, markets, regions=None, bookmakers=None):
        if "double_double" in markets or "triple_double" in markets:
            return self.milestone_response
        return self.prop_response


def test_add_player_prop_legs_fetches_only_for_the_given_match():
    player_box = pd.DataFrame({
        "player_id": ["p1"] * 5,
        "player_name": ["Home Star"] * 5,
        "team": ["Home"] * 5,
        "opponent": ["X"] * 5,
        "date": pd.date_range("2025-01-01", periods=5, freq="3D"),
        "points": [16, 18, 20, 14, 19],
        "rebounds": [4, 5, 3, 6, 4],
        "assists": [3, 2, 4, 3, 5],
        "fg3m": [2, 1, 3, 0, 2],
        "steals": [1, 0, 1, 2, 1],
        "blocks": [0, 0, 1, 0, 0],
        "turnovers": [2, 1, 2, 3, 1],
    })
    from wnba.model.player_model import compute_opponent_factors
    opponent_factors = compute_opponent_factors(player_box)

    prop_response = {"bookmakers": [{"key": "fanduel", "markets": [{"key": "player_points", "outcomes": [
        {"name": "Over", "description": "Home Star", "price": 1.9, "point": 15.5},
        {"name": "Under", "description": "Home Star", "price": 1.9, "point": 15.5},
    ]}]}]}
    milestone_response = {"bookmakers": []}

    match = MatchOptions("Home", "Away", "2026-07-05", legs=[], event_id="evt1")
    client = _PropsStubClient(prop_response, milestone_response)
    updated = add_player_prop_legs(match, player_box, opponent_factors, client=client)

    prop_legs = [l for l in updated.legs if l.market_type == "prop"]
    assert len(prop_legs) == 2
    assert {l.pick for l in prop_legs} == {"over", "under"}
    assert all(l.player_name == "Home Star" and l.stat == "points" for l in prop_legs)


def test_add_player_prop_legs_no_op_without_event_id():
    match = MatchOptions("Home", "Away", "2026-07-05", legs=[])
    updated = add_player_prop_legs(match, pd.DataFrame(), pd.DataFrame())
    assert updated is match


def test_find_best_combo_never_picks_negative_ev_leg_when_a_positive_one_exists():
    m1 = MatchOptions("A", "B", "2026-07-05", [
        _leg("A", "B", "h2h", "home", 0.5, 1.6, market_prob=0.6),  # ev negative
        _leg("A", "B", "h2h", "away", 0.5, 2.5, market_prob=0.4),  # ev positive
    ])
    m2 = MatchOptions("C", "D", "2026-07-05", [
        _leg("C", "D", "h2h", "home", 0.5, 2.0, market_prob=0.5),
    ])
    combo = find_best_combo([m1, m2], target_payout=3.2)
    m1_leg = next(leg for leg in combo if leg.home_team == "A")
    assert m1_leg.ev > 0
