from wcwinner.betbuilder import Leg, combine_legs


def _leg(home, away, pick, model_prob, odds, market_prob=None):
    return Leg(home, away, "2026-07-05", "h2h", pick, None, model_prob, odds, market_prob, market_prob is not None)


def test_combine_legs_payout_and_profit():
    legs = [_leg("A", "B", "home", 0.5, 2.0, market_prob=0.4), _leg("C", "D", "home", 0.5, 2.5, market_prob=0.4)]
    result = combine_legs(legs, stake=20.0)
    assert abs(result.combined_odds - 5.0) < 1e-9
    assert abs(result.payout - 100.0) < 1e-9
    assert abs(result.profit - 80.0) < 1e-9


def test_combine_legs_combined_probability_is_product():
    legs = [_leg("A", "B", "home", 0.6, 2.0), _leg("C", "D", "home", 0.5, 2.0)]
    result = combine_legs(legs, stake=10.0)
    assert abs(result.combined_model_prob - 0.3) < 1e-9


def test_combine_legs_single_leg():
    legs = [_leg("A", "B", "home", 0.7, 1.5)]
    result = combine_legs(legs, stake=10.0)
    assert len(result.legs) == 1
    assert abs(result.combined_odds - 1.5) < 1e-9


def test_combine_legs_returns_none_for_empty_list():
    assert combine_legs([]) is None


def test_leg_ev_and_pick_label_by_market_type():
    h2h_leg = Leg("Spain", "France", "2026-07-10", "h2h", "home", None, 0.6, 2.0, 0.55, True)
    assert abs(h2h_leg.ev - 0.2) < 1e-9
    assert h2h_leg.pick_label == "Spain to win"

    totals_leg = Leg("Spain", "France", "2026-07-10", "totals", "over", 2.5, 0.55, 1.9, 0.5, True)
    assert totals_leg.pick_label == "Over 2.5 goals"

    spread_leg = Leg("Spain", "France", "2026-07-10", "spread", "home", 0.5, 0.6, 1.9, 0.5, True)
    assert spread_leg.pick_label == "Spain +0.5"


def test_leg_edge_is_none_without_market_data():
    leg = _leg("A", "B", "home", 0.5, 2.0, market_prob=None)
    assert leg.edge is None
    assert leg.ev == 0.0  # fair odds (1/model_prob) always give exactly zero EV
