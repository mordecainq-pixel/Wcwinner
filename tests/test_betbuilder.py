from wcwinner.betbuilder import Leg, MatchOptions, combine_legs, find_best_combo


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


def test_find_best_combo_scoped_to_chosen_matches_only():
    m1 = MatchOptions("A", "B", "2026-07-05", [
        _leg("A", "B", "home", 0.6, 1.5, market_prob=0.6),
        _leg("A", "B", "away", 0.3, 3.0, market_prob=0.3),
    ])
    m2 = MatchOptions("C", "D", "2026-07-05", [
        _leg("C", "D", "home", 0.5, 2.0, market_prob=0.5),
        _leg("C", "D", "away", 0.4, 2.4, market_prob=0.4),
    ])
    # Combos: 1.5*2.0=3.0, 1.5*2.4=3.6, 3.0*2.0=6.0, 3.0*2.4=7.2 -- 3.0 is exact.
    combo = find_best_combo([m1, m2], target_payout=3.0)
    assert len(combo) == 2
    assert abs(combo[0].odds * combo[1].odds - 3.0) < 1e-9


def test_find_best_combo_prefers_higher_probability_on_ties():
    # Two combos tie exactly on distance-to-target (both hit 4.0 dead on);
    # the one with higher combined model probability should win.
    m1 = MatchOptions("A", "B", "2026-07-05", [
        _leg("A", "B", "home", 0.8, 2.0, market_prob=0.8),  # safer
        _leg("A", "B", "away", 0.1, 8.0, market_prob=0.1),  # longshot, same product w/ 0.5x partner
    ])
    m2 = MatchOptions("C", "D", "2026-07-05", [
        _leg("C", "D", "home", 0.5, 2.0, market_prob=0.5),
        _leg("C", "D", "away", 0.2, 0.5, market_prob=0.2),
    ])
    combo = find_best_combo([m1, m2], target_payout=4.0)
    combined_prob = combo[0].model_prob * combo[1].model_prob
    # The safe 2.0 x 2.0 = 4.0 combo (prob 0.8*0.5=0.4) beats the 8.0 x 0.5 = 4.0
    # combo (prob 0.1*0.2=0.02), even though both hit the target exactly.
    assert abs(combined_prob - 0.4) < 1e-9


def test_find_best_combo_never_picks_negative_ev_leg_when_a_positive_one_exists():
    m1 = MatchOptions("A", "B", "2026-07-05", [
        _leg("A", "B", "home", 0.5, 1.6, market_prob=0.6),  # ev = 0.5*1.6-1 = -0.20 (negative)
        _leg("A", "B", "away", 0.5, 2.5, market_prob=0.4),  # ev = 0.5*2.5-1 = 0.25 (positive)
    ])
    m2 = MatchOptions("C", "D", "2026-07-05", [
        _leg("C", "D", "home", 0.5, 2.0, market_prob=0.5),  # ev = 0.0 (neutral, not negative)
    ])
    # Target chosen so the negative-EV leg (1.6) combined with 2.0 = 3.2 would be
    # a closer fit than the positive-EV leg (2.5) combined with 2.0 = 5.0.
    combo = find_best_combo([m1, m2], target_payout=3.2)
    m1_leg = next(leg for leg in combo if leg.home_team == "A")
    assert m1_leg.ev > 0  # must prefer the positive-EV leg even though it fits worse


def test_leg_reasoning_mentions_model_and_market_probabilities():
    leg = _leg("A", "B", "home", 0.6, 2.0, market_prob=0.5)
    assert "60%" in leg.reasoning
    assert "50%" in leg.reasoning

    no_market_leg = _leg("A", "B", "home", 0.6, 2.0, market_prob=None)
    assert "no market data" in no_market_leg.reasoning
