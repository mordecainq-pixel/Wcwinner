"""Self-calculated Elo ratings from historical WNBA results.

Basketball has no soccer-style "friendly vs. World Cup" tournament tiers to
weight by, so instead follows the standard basketball-Elo convention (the
same shape FiveThirtyEight's public NBA Elo model used): a margin-of-victory
multiplier that rewards decisive wins but damps runaway blowout swings,
especially against much weaker opponents, plus a flat playoff K-factor bump
since playoff results matter more than they would in April. No draws exist
in basketball, which simplifies the update vs. the soccer version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wnba.config import ELO_BASE_K, ELO_HOME_ADVANTAGE, ELO_INITIAL_RATING, ELO_PLAYOFF_K_MULTIPLIER


def expected_score(rating_diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-rating_diff / 400))


def mov_multiplier(margin: int, elo_diff: float) -> float:
    """Dampens the rating swing from blowouts (especially lopsided ones
    against a much weaker-rated opponent) relative to just using raw margin,
    while still rewarding a decisive win more than a buzzer-beater.
    """
    return float(np.log(abs(margin) + 1) * (2.2 / (0.001 * abs(elo_diff) + 2.2)))


def compute_elo_history(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replay full history chronologically. Returns (results with
    home_elo_pre/away_elo_pre columns, final rating dict).
    """
    ratings: dict[str, float] = {}
    home_elo_pre, away_elo_pre = [], []

    for row in results.itertuples(index=False):
        home, away = row.home_team, row.away_team
        r_home = ratings.setdefault(home, ELO_INITIAL_RATING)
        r_away = ratings.setdefault(away, ELO_INITIAL_RATING)
        home_elo_pre.append(r_home)
        away_elo_pre.append(r_away)

        home_adv = 0 if row.neutral else ELO_HOME_ADVANTAGE
        expected_home = expected_score((r_home + home_adv) - r_away)

        margin = row.home_score - row.away_score
        actual_home = 1.0 if margin > 0 else 0.0  # no draws in basketball

        k = ELO_BASE_K * (ELO_PLAYOFF_K_MULTIPLIER if row.is_playoff else 1.0)
        delta = k * mov_multiplier(margin, r_home - r_away) * (actual_home - expected_home)

        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    out = results.copy()
    out["home_elo_pre"] = home_elo_pre
    out["away_elo_pre"] = away_elo_pre
    return out, ratings


def current_ratings(results: pd.DataFrame) -> dict[str, float]:
    _, ratings = compute_elo_history(results)
    return ratings
