from wcwinner.betbuilder import Leg, build_parlay


def _leg(home, away, pick, model_prob, odds, market_prob=None):
    return Leg(home, away, "2026-07-05", "h2h", pick, None, model_prob, odds, market_prob, market_prob is not None)


def test_build_parlay_hits_target_closely_with_two_legs():
    legs = [
        _leg("A", "B", "home", 0.60, 2.0, market_prob=0.55),  # positive EV: 0.6*2.0-1=0.2
        _leg("C", "D", "home", 0.50, 2.5, market_prob=0.45),  # positive EV: 0.5*2.5-1=0.25
        _leg("E", "F", "home", 0.80, 1.2, market_prob=0.85),  # negative EV: 0.8*1.2-1=-0.04
    ]
    result = build_parlay(legs, target_payout=5.0, stake=10.0, n_legs=2)
    assert result is not None
    assert len(result.legs) == 2
    # The two positive-EV legs multiply to 5.0 exactly; the negative-EV leg
    # should be excluded since there are enough EV-positive legs to satisfy n_legs.
    assert abs(result.combined_odds - 5.0) < 1e-9
    assert not result.used_non_edge_legs
    assert all(leg.market_prob is None or leg.ev > 0 for leg in result.legs)


def test_build_parlay_falls_back_when_not_enough_positive_ev_legs():
    legs = [
        _leg("A", "B", "home", 0.60, 2.0, market_prob=0.55),  # only 1 positive-EV leg
        _leg("C", "D", "home", 0.80, 1.1, market_prob=0.85),  # ev = 0.8*1.1-1 = -0.12
        _leg("E", "F", "home", 0.70, 1.2, market_prob=0.9),  # ev = 0.7*1.2-1 = -0.16
    ]
    result = build_parlay(legs, target_payout=3.0, stake=10.0, n_legs=3)
    assert result is not None
    assert len(result.legs) == 3
    assert result.used_non_edge_legs


def test_build_parlay_respects_no_market_data_legs_as_neutral():
    legs = [
        _leg("A", "B", "home", 0.5, 2.0, market_prob=None),
        _leg("C", "D", "home", 0.5, 2.0, market_prob=None),
    ]
    result = build_parlay(legs, target_payout=4.0, stake=10.0, n_legs=2)
    assert result is not None
    assert abs(result.combined_odds - 4.0) < 1e-9
    assert not result.used_non_edge_legs  # no-market-data legs count as eligible


def test_build_parlay_auto_picks_leg_count_when_unspecified():
    legs = [
        _leg("A", "B", "home", 0.6, 2.0, market_prob=0.5),
        _leg("C", "D", "home", 0.6, 2.0, market_prob=0.5),
        _leg("E", "F", "home", 0.6, 2.0, market_prob=0.5),
    ]
    result = build_parlay(legs, target_payout=4.0, stake=10.0, n_legs=None)
    assert result is not None
    # 2 legs of 2.0x odds each hits the target exactly; should be preferred
    # over 1 or 3 legs which would overshoot/undershoot.
    assert len(result.legs) == 2


def test_build_parlay_payout_and_profit():
    legs = [_leg("A", "B", "home", 0.5, 2.0, market_prob=0.4), _leg("C", "D", "home", 0.5, 2.5, market_prob=0.4)]
    result = build_parlay(legs, target_payout=5.0, stake=20.0, n_legs=2)
    assert abs(result.payout - 100.0) < 1e-9
    assert abs(result.profit - 80.0) < 1e-9


def test_build_parlay_returns_none_for_empty_legs():
    assert build_parlay([], target_payout=5.0) is None


def test_leg_ev_and_pick_label_by_market_type():
    h2h_leg = Leg("Spain", "France", "2026-07-10", "h2h", "home", None, 0.6, 2.0, 0.55, True)
    assert abs(h2h_leg.ev - 0.2) < 1e-9
    assert h2h_leg.pick_label == "Spain to win"

    totals_leg = Leg("Spain", "France", "2026-07-10", "totals", "over", 2.5, 0.55, 1.9, 0.5, True)
    assert totals_leg.pick_label == "Over 2.5 goals"

    spread_leg = Leg("Spain", "France", "2026-07-10", "spread", "home", 0.5, 0.6, 1.9, 0.5, True)
    assert spread_leg.pick_label == "Spain +0.5"
