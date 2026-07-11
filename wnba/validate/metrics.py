"""Log-loss and Brier score for binary (home win / away win) forecasts --
basketball has no draws, so this is simpler than wcwinner's 3-outcome version.
"""
from __future__ import annotations

import numpy as np


def log_loss(p_home_win: np.ndarray, home_won: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(np.where(home_won, p_home_win, 1 - p_home_win), eps, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(p_home_win: np.ndarray, home_won: np.ndarray) -> float:
    return float(np.mean((p_home_win - home_won.astype(float)) ** 2))
