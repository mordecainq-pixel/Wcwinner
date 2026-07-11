"""Calibration check for player-stat projections via the probability
integral transform (PIT): for each held-out player-game, simulate that
player's line using only games strictly before it, then find where the
ACTUAL result falls within the simulated distribution (0 = below every
simulated draw, 1 = above every one). If the simulation is well-calibrated,
PIT values should be uniformly distributed across [0, 1] -- clumping near
0.5 means the simulated spread is too wide, clumping near 0/1 means it's too
narrow (actuals routinely fall outside the simulated range).

Same spirit as the team model's calibration table
(validate/backtest.py + config.py's CALIBRATION_A/B writeup), adapted for a
continuous target instead of a binary win/loss outcome.

Ties are randomized (`below + U(0,1) * tied`), not just `mean(sim <= actual)`:
low-count stats like blocks have a lot of exact-zero mass in both the
simulated draws and the actual result, and a naive "<=" comparison folds
every tied zero into the "at or below" bucket, which drags mean PIT toward
1.0 for reasons that have nothing to do with calibration quality. This was
caught by running it on real backtest data -- raw mean PIT for blocks came
back 0.77 (looked badly miscalibrated), and dropped to ~0.49 once ties were
randomized, while points/rebounds (barely any exact ties) barely moved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wnba.model.player_model import STAT_COLUMNS, simulate_player_games


def pit_values(
    player_box: pd.DataFrame,
    opponent_factors: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    stat_cols: list[str] = STAT_COLUMNS,
    n_sims: int = 500,
    min_prior_games: int = 5,
    max_test_games: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    cutoff_date = pd.Timestamp(cutoff_date)
    rng = rng or np.random.default_rng(0)
    test_games = player_box[player_box["date"] > cutoff_date]
    if max_test_games and len(test_games) > max_test_games:
        test_games = test_games.sample(n=max_test_games, random_state=0)

    records = []
    for row in test_games.itertuples(index=False):
        prior = player_box[(player_box["player_id"] == row.player_id) & (player_box["date"] < row.date)]
        if len(prior) < min_prior_games:
            continue
        sim = simulate_player_games(
            player_box, row.player_id, row.opponent, opponent_factors,
            stat_cols=stat_cols, n_sims=n_sims, as_of=row.date - pd.Timedelta(days=1), rng=rng,
        )
        record = {"player_id": row.player_id, "date": row.date}
        for stat in stat_cols:
            actual = getattr(row, stat)
            below = (sim[stat] < actual).mean()
            tied = (sim[stat] == actual).mean()
            record[stat] = float(below + rng.uniform() * tied)
        records.append(record)

    return pd.DataFrame(records)


def summarize_pit(pit_df: pd.DataFrame, stat_cols: list[str] = STAT_COLUMNS) -> pd.DataFrame:
    """Mean PIT (should be ~0.5) and fraction landing in each decile
    (should be ~10% each if well-calibrated) per stat.
    """
    rows = []
    for stat in stat_cols:
        vals = pit_df[stat].dropna()
        decile_counts = pd.cut(vals, bins=np.linspace(0, 1, 11), include_lowest=True).value_counts(normalize=True).sort_index()
        rows.append({
            "stat": stat,
            "n": len(vals),
            "mean_pit": vals.mean(),
            "min_decile_pct": decile_counts.min() * 100,
            "max_decile_pct": decile_counts.max() * 100,
        })
    return pd.DataFrame(rows)
