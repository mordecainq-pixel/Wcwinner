import numpy as np
import pandas as pd
import pytest

from wnba.model.player_model import (
    STAT_COLUMNS,
    compute_opponent_factors,
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
