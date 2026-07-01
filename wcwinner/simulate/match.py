"""Single-matchup prediction output: expected goals, full scoreline
probability matrix, win/draw/loss %, both-teams-to-score %, and — for
knockout matches — advancement probability including the ET/penalty tiebreak.
"""
from __future__ import annotations

import numpy as np

from wcwinner.model.dixon_coles import DixonColesModel, build_score_matrix
from wcwinner.model.knockout import knockout_outcome_probs


def predict_match(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    neutral: bool = True,
    knockout: bool = False,
    elo_ratings: dict[str, float] | None = None,
    max_goals: int = 8,
    home_strength_multiplier: float = 1.0,
    away_strength_multiplier: float = 1.0,
) -> dict:
    """`home_strength_multiplier` / `away_strength_multiplier` are a manual,
    optional knob for squad news (injuries/suspensions) -- e.g. 0.85 to shave
    15% off a team's expected goals for a missing striker. This is NOT
    automated: there's no reliable free feed for team news, so it defaults to
    1.0 (no adjustment) and only moves if a caller explicitly sets it.
    """
    lam, mu = model.expected_goals(home_team, away_team, neutral)
    lam *= home_strength_multiplier
    mu *= away_strength_multiplier
    matrix = build_score_matrix(lam, mu, model.rho, max_goals)

    p_home_win = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away_win = float(np.triu(matrix, 1).sum())

    i = np.arange(matrix.shape[0])[:, None]
    j = np.arange(matrix.shape[1])[None, :]
    p_btts = float(matrix[(i > 0) & (j > 0)].sum())

    most_likely_idx = np.unravel_index(np.argmax(matrix), matrix.shape)

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "neutral": neutral,
        "expected_goals_home": lam,
        "expected_goals_away": mu,
        "score_matrix": matrix,
        "p_home_win": p_home_win,
        "p_draw": p_draw,
        "p_away_win": p_away_win,
        "p_btts": p_btts,
        "most_likely_score": most_likely_idx,
        "most_likely_score_prob": float(matrix[most_likely_idx]),
    }

    if knockout:
        if elo_ratings is None:
            raise ValueError("elo_ratings is required for knockout-match tiebreak probabilities")
        result["knockout"] = knockout_outcome_probs(
            model, home_team, away_team, elo_ratings, neutral, max_goals,
            home_strength_multiplier, away_strength_multiplier,
        )

    return result
