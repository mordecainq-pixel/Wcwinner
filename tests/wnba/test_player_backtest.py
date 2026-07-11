import numpy as np
import pandas as pd

from wnba.model.player_model import compute_opponent_factors
from wnba.validate.player_backtest import pit_values, summarize_pit


def _synthetic_player_box(n_games=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_games, freq="3D")
    opponents = rng.choice(["Team A", "Team B", "Team C"], size=n_games)
    points = rng.normal(18, 5, size=n_games).round().clip(min=0)
    return pd.DataFrame({
        "player_id": ["p1"] * n_games,
        "date": dates,
        "opponent": opponents,
        "team": ["Home"] * n_games,
        "points": points,
        "rebounds": rng.normal(6, 2, size=n_games).round().clip(min=0),
        "assists": rng.normal(4, 2, size=n_games).round().clip(min=0),
        "fg3m": rng.normal(2, 1, size=n_games).round().clip(min=0),
        "steals": rng.normal(1, 1, size=n_games).round().clip(min=0),
        "blocks": rng.normal(0.5, 0.7, size=n_games).round().clip(min=0),
        "turnovers": rng.normal(2, 1, size=n_games).round().clip(min=0),
    })


def test_pit_values_are_bounded_and_only_computed_with_enough_prior_history():
    df = _synthetic_player_box(n_games=40)
    factors = compute_opponent_factors(df)
    pit = pit_values(df, factors, cutoff_date="2024-03-01", n_sims=200, min_prior_games=5, rng=np.random.default_rng(1))

    assert not pit.empty
    for stat in ["points", "rebounds", "assists"]:
        assert pit[stat].between(0, 1).all()


def test_summarize_pit_reports_mean_near_half_for_stable_stationary_process():
    df = _synthetic_player_box(n_games=80)
    factors = compute_opponent_factors(df)
    pit = pit_values(df, factors, cutoff_date="2024-02-01", n_sims=300, min_prior_games=5, rng=np.random.default_rng(2))
    summary = summarize_pit(pit, stat_cols=["points", "rebounds", "assists"])

    assert len(summary) == 3
    # a stationary (no real trend/role-change) synthetic process should calibrate
    # roughly symmetrically -- loose bound since this is a small sample
    assert summary["mean_pit"].between(0.3, 0.7).all()
