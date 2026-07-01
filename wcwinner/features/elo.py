"""Self-calculated Elo ratings from the full 1872-2026 match history.

Follows the published World Football Elo Ratings methodology
(eloratings.net/about) rather than plain chess-style Elo: match importance
scales the K-factor, and the rating swing is scaled further by margin of
victory. Computed from scratch here (not pulled from a third-party feed) so
it updates automatically as new results come in and stays internally
consistent with the rest of the model.
"""
from __future__ import annotations

import pandas as pd

from wcwinner.config import (
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL_RATING,
    ELO_K_CONTINENTAL_OR_INTERCONFEDERATION,
    ELO_K_FRIENDLY,
    ELO_K_OTHER_TOURNAMENT,
    ELO_K_QUALIFIERS,
    ELO_K_WORLD_CUP_FINALS,
)

# Major continental / intercontinental championships (not their qualifiers,
# which are matched separately below).
_CONTINENTAL_MAJOR = {
    "uefa euro",
    "copa américa",
    "copa america",
    "african cup of nations",
    "afc asian cup",
    "concacaf championship",
    "gold cup",
    "confederations cup",
    "oceania nations cup",
    "uefa nations league",
    "concacaf nations league",
}


def tournament_k_factor(tournament: str) -> int:
    name = tournament.lower().strip()
    if "qualification" in name or "qualifying" in name:
        return ELO_K_QUALIFIERS
    if name == "friendly":
        return ELO_K_FRIENDLY
    if name == "fifa world cup":
        return ELO_K_WORLD_CUP_FINALS
    if name in _CONTINENTAL_MAJOR:
        return ELO_K_CONTINENTAL_OR_INTERCONFEDERATION
    return ELO_K_OTHER_TOURNAMENT


def goal_diff_weight(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def expected_score(rating_diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-rating_diff / 400))


def compute_elo_history(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replay full history chronologically, updating Elo after each played match.

    Returns (results with home_elo_pre/away_elo_pre columns added, final
    rating dict). Pre-match ratings are the *raw* ratings (home advantage is
    applied only inside the expected-score calc, not stored in the rating
    itself), so they're directly usable as a symmetric feature elsewhere.
    """
    ratings: dict[str, float] = {}
    home_elo_pre = []
    away_elo_pre = []

    for row in results.itertuples(index=False):
        home, away = row.home_team, row.away_team
        r_home = ratings.setdefault(home, ELO_INITIAL_RATING)
        r_away = ratings.setdefault(away, ELO_INITIAL_RATING)
        home_elo_pre.append(r_home)
        away_elo_pre.append(r_away)

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue  # unplayed fixture, no rating update

        home_advantage = 0 if row.neutral else ELO_HOME_ADVANTAGE
        effective_diff = (r_home + home_advantage) - r_away
        expected_home = expected_score(effective_diff)

        goal_diff = row.home_score - row.away_score
        if goal_diff > 0:
            actual_home = 1.0
        elif goal_diff == 0:
            actual_home = 0.5
        else:
            actual_home = 0.0

        k = tournament_k_factor(row.tournament)
        delta = k * goal_diff_weight(goal_diff) * (actual_home - expected_home)

        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    out = results.copy()
    out["home_elo_pre"] = home_elo_pre
    out["away_elo_pre"] = away_elo_pre
    return out, ratings


def current_ratings(results: pd.DataFrame) -> dict[str, float]:
    _, ratings = compute_elo_history(results)
    return ratings
