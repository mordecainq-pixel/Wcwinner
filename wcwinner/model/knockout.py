"""Knockout-match resolution: regulation result via Dixon-Coles, then a
tilted coin flip for extra time + penalties when regulation ends level.

Real shootouts and extra-time golden moments are close to random relative to
run-of-play scoring rates, which is why this doesn't just keep sampling from
the Poisson model until someone leads — that would implicitly assume ET goal
rates match normal-time rates, which they don't (both teams sit deeper, fewer
clear chances). A small Elo-based tilt captures the modest real edge stronger
squads have in a shootout (depth, penalty-taking experience) without
overstating it.
"""
from __future__ import annotations

import numpy as np

from wcwinner.config import KNOCKOUT_TIEBREAK_ELO_SCALE, KNOCKOUT_TIEBREAK_ELO_WEIGHT
from wcwinner.model.dixon_coles import DixonColesModel


def tiebreak_home_win_prob(home_team: str, away_team: str, elo_ratings: dict[str, float]) -> float:
    elo_diff = elo_ratings.get(home_team, 1500.0) - elo_ratings.get(away_team, 1500.0)
    tilt = KNOCKOUT_TIEBREAK_ELO_WEIGHT * np.tanh(elo_diff / KNOCKOUT_TIEBREAK_ELO_SCALE)
    return float(0.5 + tilt)


def knockout_outcome_probs(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    elo_ratings: dict[str, float],
    neutral: bool = True,
    max_goals: int = 8,
) -> dict:
    """Probability the home/away team advances, decomposing regulation win vs.
    a draw sent to the ET/penalty tiebreak.
    """
    matrix = model.score_matrix(home_team, away_team, neutral=neutral, max_goals=max_goals)
    p_home_reg_win = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away_reg_win = float(np.triu(matrix, 1).sum())

    p_home_tiebreak = tiebreak_home_win_prob(home_team, away_team, elo_ratings)

    p_home_advances = p_home_reg_win + p_draw * p_home_tiebreak
    p_away_advances = p_away_reg_win + p_draw * (1 - p_home_tiebreak)

    return {
        "p_home_regulation_win": p_home_reg_win,
        "p_draw_regulation": p_draw,
        "p_away_regulation_win": p_away_reg_win,
        "p_home_tiebreak_win": p_home_tiebreak,
        "p_home_advances": p_home_advances,
        "p_away_advances": p_away_advances,
    }


def sample_knockout_winner(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    elo_ratings: dict[str, float],
    rng: np.random.Generator,
    neutral: bool = True,
    max_goals: int = 8,
) -> str:
    """Draw one Monte Carlo outcome: sample a scoreline, and if it's a draw,
    flip the tilted tiebreak coin.
    """
    matrix = model.score_matrix(home_team, away_team, neutral=neutral, max_goals=max_goals)
    flat = matrix.ravel()
    idx = rng.choice(len(flat), p=flat)
    home_goals, away_goals = divmod(idx, matrix.shape[1])

    if home_goals > away_goals:
        return home_team
    if away_goals > home_goals:
        return away_team

    p_home_tiebreak = tiebreak_home_win_prob(home_team, away_team, elo_ratings)
    return home_team if rng.random() < p_home_tiebreak else away_team
