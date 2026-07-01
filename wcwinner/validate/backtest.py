"""Backtest: fit on data through a cutoff date, evaluate on real matches
after it, and compare against a naive "higher Elo always wins" baseline.

The baseline converts the Elo rating gap into a win probability via the
standard Elo expected-score formula and assigns the historical league-wide
draw rate as a flat P(draw) — deliberately simple, since the point is to show
how much the fitted Dixon-Coles attack/defense structure adds over "just use
the rating gap."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wcwinner.features.elo import compute_elo_history, expected_score
from wcwinner.model.dixon_coles import fit
from wcwinner.simulate.match import predict_match
from wcwinner.validate.metrics import brier_score, log_loss, outcome_index


def elo_only_baseline_probs(elo_home: float, elo_away: float, neutral: bool, home_advantage: float, draw_rate: float) -> tuple[float, float, float]:
    adv = 0 if neutral else home_advantage
    p_home_no_draw = expected_score((elo_home + adv) - elo_away)
    p_home = p_home_no_draw * (1 - draw_rate)
    p_away = (1 - p_home_no_draw) * (1 - draw_rate)
    return p_home, draw_rate, p_away


def run_backtest(results: pd.DataFrame, cutoff_date: str | pd.Timestamp, max_test_matches: int | None = None) -> dict:
    cutoff_date = pd.Timestamp(cutoff_date)
    train = results[results["date"] <= cutoff_date].copy()
    test = results[
        (results["date"] > cutoff_date) & results["home_score"].notna() & results["away_score"].notna()
    ].copy()
    if max_test_matches:
        test = test.head(max_test_matches)

    _, elo_ratings = compute_elo_history(train)
    model = fit(train, elo_ratings, as_of=cutoff_date)

    from wcwinner.config import ELO_HOME_ADVANTAGE

    draw_rate = float((train["home_score"] == train["away_score"]).mean())

    model_probs, baseline_probs, true_idx = [], [], []
    for row in test.itertuples(index=False):
        pred = predict_match(model, row.home_team, row.away_team, neutral=row.neutral)
        model_probs.append([pred["p_home_win"], pred["p_draw"], pred["p_away_win"]])

        eh = elo_ratings.get(row.home_team, 1500.0)
        ea = elo_ratings.get(row.away_team, 1500.0)
        baseline_probs.append(elo_only_baseline_probs(eh, ea, row.neutral, ELO_HOME_ADVANTAGE, draw_rate))

        true_idx.append(outcome_index(row.home_score, row.away_score))

    model_probs = np.array(model_probs)
    baseline_probs = np.array(baseline_probs)
    true_idx = np.array(true_idx)

    return {
        "cutoff_date": cutoff_date,
        "n_train_matches": len(train),
        "n_test_matches": len(test),
        "model": model,
        "elo_ratings": elo_ratings,
        "model_probs": model_probs,
        "baseline_probs": baseline_probs,
        "true_idx": true_idx,
        "test_matches": test,
        "model_log_loss": log_loss(model_probs, true_idx),
        "model_brier": brier_score(model_probs, true_idx),
        "baseline_log_loss": log_loss(baseline_probs, true_idx),
        "baseline_brier": brier_score(baseline_probs, true_idx),
    }
