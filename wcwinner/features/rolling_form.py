"""Decay-weighted rolling form, rest-day fatigue proxy, and head-to-head
goal differential — computed causally (pre-match) for every match in order.

These are exposed for CLI/app context and validation diagnostics rather than
fed directly into the Dixon-Coles MLE: the DC fit already applies its own
time-decay over the full history plus Elo-based shrinkage for small samples
(see wcwinner.model.dixon_coles), so folding a second, differently-scoped
decay window into the same likelihood would double up on what "recent form"
already means without a clear calibration benefit. See README for the full
rationale.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from wcwinner.config import ROLLING_FORM_HALF_LIFE, ROLLING_FORM_WINDOW


def _decay_weights(n: int) -> np.ndarray:
    decay = 0.5 ** (1 / ROLLING_FORM_HALF_LIFE)
    weights = decay ** np.arange(n - 1, -1, -1)  # oldest entry first, most recent = decay**0
    return weights / weights.sum()


def compute_rolling_form(results: pd.DataFrame) -> pd.DataFrame:
    """Requires `results` sorted chronologically (as returned by
    wcwinner.data.kaggle_ingest.load_results).
    """
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_FORM_WINDOW))
    last_match_date: dict[str, pd.Timestamp] = {}
    h2h_goal_diff: dict[tuple[str, str], float] = defaultdict(float)

    cols = {
        "home_attack_form": [],
        "home_defense_form": [],
        "away_attack_form": [],
        "away_defense_form": [],
        "home_rest_days": [],
        "away_rest_days": [],
        "h2h_goal_diff": [],
    }

    def form(team: str) -> tuple[float, float]:
        entries = history[team]
        if not entries:
            return np.nan, np.nan
        weights = _decay_weights(len(entries))
        goals_for = np.array([e[0] for e in entries])
        goals_against = np.array([e[1] for e in entries])
        return float((weights * goals_for).sum()), float((weights * goals_against).sum())

    for row in results.itertuples(index=False):
        home, away, date = row.home_team, row.away_team, row.date

        h_att, h_def = form(home)
        a_att, a_def = form(away)
        cols["home_attack_form"].append(h_att)
        cols["home_defense_form"].append(h_def)
        cols["away_attack_form"].append(a_att)
        cols["away_defense_form"].append(a_def)

        cols["home_rest_days"].append((date - last_match_date[home]).days if home in last_match_date else np.nan)
        cols["away_rest_days"].append((date - last_match_date[away]).days if away in last_match_date else np.nan)

        cols["h2h_goal_diff"].append(h2h_goal_diff[(home, away)])

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue  # unplayed fixture: features recorded, no state update

        history[home].append((row.home_score, row.away_score))
        history[away].append((row.away_score, row.home_score))
        last_match_date[home] = date
        last_match_date[away] = date

        goal_diff = row.home_score - row.away_score
        h2h_goal_diff[(home, away)] += goal_diff
        h2h_goal_diff[(away, home)] -= goal_diff

    out = results.copy()
    for name, values in cols.items():
        out[name] = values
    return out
