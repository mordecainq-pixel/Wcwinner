"""Log-loss and (multi-class) Brier score for 3-outcome (home/draw/away)
probabilistic forecasts. Implemented directly rather than pulled from
scikit-learn to avoid a heavy dependency for two formulas this small.
"""
from __future__ import annotations

import numpy as np

HOME, DRAW, AWAY = 0, 1, 2


def outcome_index(home_score: float, away_score: float) -> int:
    if home_score > away_score:
        return HOME
    if home_score == away_score:
        return DRAW
    return AWAY


def log_loss(probs: np.ndarray, true_idx: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(probs[np.arange(len(true_idx)), true_idx], eps, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(probs: np.ndarray, true_idx: np.ndarray) -> float:
    true_onehot = np.eye(probs.shape[1])[true_idx]
    return float(np.mean(np.sum((probs - true_onehot) ** 2, axis=1)))
