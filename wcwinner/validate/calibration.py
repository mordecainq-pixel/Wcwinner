"""Reliability check: do the model's "70% win" calls actually land ~70% of
the time? Bins predictions by predicted probability and compares against the
observed frequency in each bin, separately for each outcome type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(predicted_probs: np.ndarray, actual_binary: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(predicted_probs, bin_edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_low": bin_edges[b],
                "bin_high": bin_edges[b + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(predicted_probs[mask].mean()),
                "observed_frequency": float(actual_binary[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_report(model_probs: np.ndarray, true_idx: np.ndarray, n_bins: int = 10) -> dict[str, pd.DataFrame]:
    outcomes = {"home_win": 0, "draw": 1, "away_win": 2}
    return {
        name: reliability_table(model_probs[:, idx], (true_idx == idx).astype(float), n_bins)
        for name, idx in outcomes.items()
    }
