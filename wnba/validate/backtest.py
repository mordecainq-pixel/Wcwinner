"""Backtest: fit through a cutoff, evaluate on real held-out games, compare
against a naive "higher Elo always wins" baseline. Same methodology as
wcwinner's soccer backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wnba.features.elo import compute_elo_history, expected_score
from wnba.model.margin_model import fit
from wnba.validate.metrics import brier_score, log_loss


def run_backtest(results: pd.DataFrame, cutoff_date: str | pd.Timestamp, max_test_games: int | None = None) -> dict:
    cutoff_date = pd.Timestamp(cutoff_date)
    train = results[results["date"] <= cutoff_date].copy()
    test = results[results["date"] > cutoff_date].copy()
    if max_test_games:
        test = test.head(max_test_games)

    _, elo_ratings = compute_elo_history(train)
    model = fit(train, elo_ratings, as_of=cutoff_date)

    from wnba.config import ELO_HOME_ADVANTAGE

    model_probs, baseline_probs, home_won = [], [], []
    for row in test.itertuples(index=False):
        model_probs.append(model.win_probability(row.home_team, row.away_team, row.neutral))

        eh = elo_ratings.get(row.home_team, 1500.0)
        ea = elo_ratings.get(row.away_team, 1500.0)
        adv = 0 if row.neutral else ELO_HOME_ADVANTAGE
        baseline_probs.append(expected_score((eh + adv) - ea))

        home_won.append(row.home_score > row.away_score)

    model_probs = np.array(model_probs)
    baseline_probs = np.array(baseline_probs)
    home_won = np.array(home_won)

    return {
        "cutoff_date": cutoff_date,
        "n_train": len(train),
        "n_test": len(test),
        "model": model,
        "elo_ratings": elo_ratings,
        "model_log_loss": log_loss(model_probs, home_won),
        "model_brier": brier_score(model_probs, home_won),
        "baseline_log_loss": log_loss(baseline_probs, home_won),
        "baseline_brier": brier_score(baseline_probs, home_won),
        "model_probs": model_probs,
        "home_won": home_won,
    }
