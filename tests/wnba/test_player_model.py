import numpy as np
import pandas as pd
import pytest

from wnba.model.player_model import (
    STAT_COLUMNS,
    compute_opponent_factors,
    compute_out_player_adjustment,
    prob_double_double,
    prob_over,
    prob_triple_double,
    simulate_player_games,
)


def _rows(player_id, opponent, points, n, start_date="2025-01-01"):
    dates = pd.date_range(start_date, periods=n, freq="3D")
    return pd.DataFrame({
        "player_id": [player_id] * n,
        "date": dates,
        "opponent": [opponent] * n,
        "points": [points] * n,
        "rebounds": [0] * n,
        "assists": [0] * n,
        "fg3m": [0] * n,
        "steals": [0] * n,
        "blocks": [0] * n,
        "turnovers": [0] * n,
    })


def test_opponent_factors_reflect_stats_allowed_and_shrink_small_samples():
    df = pd.concat([
        _rows("p_avg", "Average", 20, 40),
        _rows("p_weak", "Weak", 30, 40),
        _rows("p_strong", "Strong", 10, 40),
        _rows("p_new", "New", 25, 2),
    ])
    factors = compute_opponent_factors(df, stat_cols=STAT_COLUMNS)

    assert factors.loc["Weak", "points"] > 1.0
    assert factors.loc["Strong", "points"] < 1.0
    assert factors.loc["Weak", "points"] > factors.loc["Strong", "points"]
    # small-sample team's factor should be shrunk close to 1.0, not the raw ~1.25
    assert abs(factors.loc["New", "points"] - 1.0) < 0.05


def test_simulate_player_games_requires_minimum_history():
    df = _rows("p1", "X", 15, 2)
    factors = compute_opponent_factors(df)
    with pytest.raises(ValueError):
        simulate_player_games(df, "p1", "Unseen", factors, n_sims=100, rng=np.random.default_rng(0))


def test_recency_weighting_favors_recent_role_change():
    early = _rows("p1", "X", 10, 20, start_date="2024-01-01")
    recent = _rows("p1", "X", 50, 5, start_date="2025-06-01")
    df = pd.concat([early, recent])
    factors = compute_opponent_factors(df)

    sim = simulate_player_games(
        df, "p1", "Unseen", factors, n_sims=5000, halflife_games=2, rng=np.random.default_rng(0),
    )
    # a short halflife should weight the recent 50-point stretch heavily,
    # well above the naive unweighted mean of ~18
    assert sim["points"].mean() > 35


def test_prob_over_and_combo_columns():
    early = _rows("p1", "X", 20, 30)
    df = early.assign(rebounds=5, assists=5)
    factors = compute_opponent_factors(df)
    sim = simulate_player_games(df, "p1", "Unseen", factors, n_sims=2000, rng=np.random.default_rng(1))

    assert prob_over(sim, "points", 10) == pytest.approx(1.0)
    assert prob_over(sim, "points", 100) == pytest.approx(0.0)
    assert prob_over(sim, "pra", 25) == pytest.approx(1.0)  # 20+5+5 = 30 > 25 always


def _rival_scenario(n_out_games=10, star_minutes=30.0):
    """Rival's own history (so star1 qualifies as a rotation player) plus
    games faced by Rival, split into games star1 played vs. missed --
    opponents score more (points=30 vs 20) in the missed games.
    """
    rival_own = pd.DataFrame({
        "player_id": ["star1"] * 20,
        "team": ["Rival"] * 20,
        "opponent": ["X"] * 20,
        "event_id": [f"r{i}" for i in range(20)],
        "date": pd.date_range("2024-01-01", periods=20, freq="3D"),
        "minutes": [star_minutes] * 20,
        "points": [15.0] * 20, "rebounds": [5.0] * 20, "assists": [3.0] * 20,
        "fg3m": [1.0] * 20, "steals": [1.0] * 20, "blocks": [0.5] * 20, "turnovers": [2.0] * 20,
    })
    present_events = [f"p{i}" for i in range(30)]
    out_events = [f"out{i}" for i in range(n_out_games)]
    faced_present = pd.DataFrame({
        "player_id": [f"opp{i}" for i in range(30)],
        "team": ["Other"] * 30,
        "opponent": ["Rival"] * 30,
        "event_id": present_events,
        "date": pd.date_range("2024-02-01", periods=30, freq="1D"),
        "minutes": [30.0] * 30,
        "points": [20.0] * 30, "rebounds": [5.0] * 30, "assists": [3.0] * 30,
        "fg3m": [1.0] * 30, "steals": [1.0] * 30, "blocks": [0.5] * 30, "turnovers": [2.0] * 30,
    })
    faced_out = pd.DataFrame({
        "player_id": [f"oppo{i}" for i in range(n_out_games)],
        "team": ["Other"] * n_out_games,
        "opponent": ["Rival"] * n_out_games,
        "event_id": out_events,
        "date": pd.date_range("2024-03-15", periods=n_out_games, freq="1D"),
        "minutes": [30.0] * n_out_games,
        "points": [30.0] * n_out_games, "rebounds": [5.0] * n_out_games, "assists": [3.0] * n_out_games,
        "fg3m": [1.0] * n_out_games, "steals": [1.0] * n_out_games, "blocks": [0.5] * n_out_games, "turnovers": [2.0] * n_out_games,
    })
    player_box = pd.concat([rival_own, faced_present, faced_out], ignore_index=True)

    avail_rows = [
        {"event_id": eid, "date": d, "team": "Rival", "player_id": "star1", "played": True}
        for eid, d in zip(present_events, faced_present["date"])
    ] + [
        {"event_id": eid, "date": d, "team": "Rival", "player_id": "star1", "played": False}
        for eid, d in zip(out_events, faced_out["date"])
    ]
    availability = pd.DataFrame(avail_rows)
    return player_box, availability


def test_out_player_adjustment_reflects_real_missed_games():
    player_box, availability = _rival_scenario(n_out_games=10)
    adj = compute_out_player_adjustment(player_box, availability, "Rival", {"star1"})
    # opponents scored more (30 vs 20) in star1's real missed games -- factor should reflect that
    assert adj["points"] > 1.05
    assert adj["points"] < 1.6  # within the configured clip band, not an extreme swing


def test_out_player_adjustment_shrinks_toward_one_with_few_missed_games():
    small_box, small_avail = _rival_scenario(n_out_games=1)
    large_box, large_avail = _rival_scenario(n_out_games=10)
    adj_small = compute_out_player_adjustment(small_box, small_avail, "Rival", {"star1"})
    adj_large = compute_out_player_adjustment(large_box, large_avail, "Rival", {"star1"})
    # same underlying per-game effect, but 1 missed game should be shrunk closer to 1.0
    # than 10 missed games -- less evidence, less confidence, not the raw ratio either time
    assert abs(adj_small["points"] - 1.0) < abs(adj_large["points"] - 1.0)


def test_out_player_adjustment_ignores_non_rotation_scratch():
    player_box, availability = _rival_scenario(n_out_games=10)
    # "bench1" has almost no track record on Rival -- shouldn't qualify
    bench_rows = pd.DataFrame({
        "player_id": ["bench1"] * 2,
        "team": ["Rival"] * 2,
        "opponent": ["X"] * 2,
        "event_id": ["b0", "b1"],
        "date": pd.date_range("2024-01-01", periods=2, freq="3D"),
        "minutes": [4.0, 6.0],
        "points": [1.0, 2.0], "rebounds": [0.0, 1.0], "assists": [0.0, 0.0],
        "fg3m": [0.0, 0.0], "steals": [0.0, 0.0], "blocks": [0.0, 0.0], "turnovers": [0.0, 0.0],
    })
    player_box = pd.concat([player_box, bench_rows], ignore_index=True)
    adj = compute_out_player_adjustment(player_box, availability, "Rival", {"bench1"})
    assert (adj == 1.0).all()


def test_out_player_adjustment_no_out_players_is_identity():
    player_box, availability = _rival_scenario()
    adj = compute_out_player_adjustment(player_box, availability, "Rival", set())
    assert (adj == 1.0).all()


def test_simulate_player_games_applies_out_player_adjustment():
    player_box, availability = _rival_scenario(n_out_games=10)
    factors = compute_opponent_factors(player_box)
    p1_rows = pd.DataFrame({
        "player_id": ["p1"] * 20,
        "date": pd.date_range("2024-04-01", periods=20, freq="1D"),
        "opponent": ["Rival"] * 20,
        "team": ["Shooter"] * 20,
        "event_id": [f"s{i}" for i in range(20)],
        "points": [20.0] * 20, "rebounds": [0.0] * 20, "assists": [0.0] * 20,
        "fg3m": [0.0] * 20, "steals": [0.0] * 20, "blocks": [0.0] * 20, "turnovers": [0.0] * 20,
    })
    player_box = pd.concat([player_box, p1_rows], ignore_index=True)

    baseline = simulate_player_games(player_box, "p1", "Rival", factors, n_sims=3000, rng=np.random.default_rng(0))
    adjusted = simulate_player_games(
        player_box, "p1", "Rival", factors, n_sims=3000, rng=np.random.default_rng(0),
        out_player_ids={"star1"}, availability=availability,
    )
    assert adjusted["points"].mean() > baseline["points"].mean()


def test_double_double_and_triple_double_probabilities():
    sim = pd.DataFrame({
        "points": [20, 20, 20, 5],
        "rebounds": [12, 3, 12, 3],
        "assists": [3, 3, 11, 3],
        "steals": [0, 0, 0, 0],
        "blocks": [0, 0, 0, 0],
    })
    # row0: pts+reb = double-double. row1: only pts = not. row2: pts+reb+ast = triple-double. row3: none.
    assert prob_double_double(sim) == pytest.approx(0.5)
    assert prob_triple_double(sim) == pytest.approx(0.25)
