from wcwinner.betbuilder import Leg, build_parlay


def _leg(home, away, pick, model_prob, odds, market_prob=None):
    return Leg(home, away, "2026-07-05", pick, model_prob, odds, market_prob, market_prob is not None)


def test_build_parlay_hits_target_closely_with_two_legs():
    legs = [
        _leg("A", "B", "home", 0.60, 2.0, market_prob=0.55),  # positive edge
        _leg("C", "D", "home", 0.50, 2.5, market_prob=0.45),  # positive edge
        _leg("E", "F", "home", 0.80, 1.2, market_prob=0.85),  # negative edge
    ]
    result = build_parlay(legs, target_payout=5.0, stake=10.0, n_legs=2)
    assert result is not None
    assert len(result.legs) == 2
    # The two positive-edge legs multiply to 5.0 exactly; the negative-edge
    # leg should be excluded since there are enough edge legs to satisfy n_legs.
    assert abs(result.combined_odds - 5.0) < 1e-9
    assert not result.used_non_edge_legs
    assert all(leg.edge is None or leg.edge > 0 for leg in result.legs)


def test_build_parlay_falls_back_when_not_enough_edge_legs():
    legs = [
        _leg("A", "B", "home", 0.60, 2.0, market_prob=0.55),  # only 1 positive-edge leg
        _leg("C", "D", "home", 0.80, 1.3, market_prob=0.85),
        _leg("E", "F", "home", 0.70, 1.4, market_prob=0.9),
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
    assert not result.used_non_edge_legs  # None-edge legs count as eligible


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
