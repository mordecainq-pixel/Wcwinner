"""Player prop projections: weighted-bootstrap Monte Carlo over a player's
own recent box scores, scaled by the opponent's "stats allowed" factor.

Why bootstrap instead of fitting a Normal/Poisson per stat: props need
combos (PRA, points+rebounds, ...) and milestones (double-double,
triple-double) that all depend on the SAME game's stats moving together --
a player's big scoring nights also tend to be their big-assist nights.
Fitting independent marginals per stat and summing them would understate
that correlation. Resampling whole historical game rows (not per-stat
values) preserves the real within-game correlation for free, and it's
naturally robust to low counts (blocks, steals) where a Normal/Poisson
choice would matter a lot but a raw empirical distribution doesn't care.

Recency weighting is in GAMES, not days, unlike the team model's day-based
decay: a player's role can change from one game to the next (new starting
job, coming back from injury, a trade), so "games since" is the more
meaningful clock than calendar time for who this player currently is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wnba.config import (
    PLAYER_FORM_HALFLIFE_GAMES,
    PLAYER_MIN_GAMES_FOR_PROJECTION,
    PLAYER_OPPONENT_SHRINKAGE_K,
    PLAYER_SIM_COUNT,
)

STAT_COLUMNS = ["points", "rebounds", "assists", "fg3m", "steals", "blocks", "turnovers"]
_DOUBLE_DOUBLE_COLUMNS = ["points", "rebounds", "assists", "steals", "blocks"]

STAT_LABELS = {
    "points": "Points", "rebounds": "Rebounds", "assists": "Assists",
    "fg3m": "3-Pointers Made", "steals": "Steals", "blocks": "Blocks", "turnovers": "Turnovers",
    "pra": "Pts+Reb+Ast", "pr": "Pts+Reb", "pa": "Pts+Ast", "ra": "Reb+Ast", "sb": "Stl+Blk",
    "double_double": "Double-Double", "triple_double": "Triple-Double",
}


def compute_opponent_factors(player_box: pd.DataFrame, stat_cols: list[str] = STAT_COLUMNS) -> pd.DataFrame:
    """For each team, how much more/less than league-average each stat runs
    for players who face them (i.e. what that team "allows"). >1 means that
    team allows more of that stat than average (weaker on that dimension),
    <1 means it suppresses it. Shrunk toward 1.0 (no effect) for teams with
    a small number of games faced, same empirical-Bayes spirit as the team
    model's Elo-blend shrinkage.
    """
    league_avg = player_box[stat_cols].mean()
    allowed = player_box.groupby("opponent")[stat_cols].mean()
    counts = player_box.groupby("opponent").size()

    raw_factor = allowed.div(league_avg, axis=1)
    shrink_weight = (counts / (counts + PLAYER_OPPONENT_SHRINKAGE_K)).reindex(raw_factor.index)
    shrunk = raw_factor.mul(shrink_weight, axis=0).add(1.0, axis=0).sub(shrink_weight, axis=0)
    return shrunk


def _recency_weights(n: int, halflife_games: float) -> np.ndarray:
    games_ago = np.arange(n)[::-1]  # row 0 (oldest) gets the largest games_ago
    weights = 0.5 ** (games_ago / halflife_games)
    return weights / weights.sum()


def simulate_player_games(
    player_box: pd.DataFrame,
    player_id: str,
    opponent_team: str,
    opponent_factors: pd.DataFrame,
    stat_cols: list[str] = STAT_COLUMNS,
    n_sims: int = PLAYER_SIM_COUNT,
    halflife_games: float = PLAYER_FORM_HALFLIFE_GAMES,
    as_of: pd.Timestamp | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Weighted-bootstrap `n_sims` simulated game lines for one player
    against one opponent. Returns a DataFrame with `stat_cols` plus derived
    combo columns (pra, pr, pa, ra, sb).
    """
    rows = player_box[player_box["player_id"] == player_id].sort_values("date")
    if as_of is not None:
        rows = rows[rows["date"] <= as_of]
    if len(rows) < PLAYER_MIN_GAMES_FOR_PROJECTION:
        raise ValueError(f"player {player_id} has only {len(rows)} games, need >= {PLAYER_MIN_GAMES_FOR_PROJECTION}")

    weights = _recency_weights(len(rows), halflife_games)
    rng = rng or np.random.default_rng()
    sample_idx = rng.choice(len(rows), size=n_sims, replace=True, p=weights)
    sampled = rows.iloc[sample_idx][stat_cols].reset_index(drop=True)

    if opponent_team in opponent_factors.index:
        factor_row = opponent_factors.loc[opponent_team, stat_cols]
    else:
        factor_row = pd.Series(1.0, index=stat_cols)  # unseen opponent (e.g. new franchise): no adjustment
    sim = sampled.mul(factor_row, axis=1)

    sim["pra"] = sim["points"] + sim["rebounds"] + sim["assists"]
    sim["pr"] = sim["points"] + sim["rebounds"]
    sim["pa"] = sim["points"] + sim["assists"]
    sim["ra"] = sim["rebounds"] + sim["assists"]
    sim["sb"] = sim["steals"] + sim["blocks"]
    return sim


def prob_over(sim: pd.DataFrame, stat: str, line: float) -> float:
    """P(stat > line) from simulated draws. `stat` may be a raw column
    (points, rebounds, ...) or a combo column (pra, pr, pa, ra, sb)."""
    return float((sim[stat] > line).mean())


def prob_double_double(sim: pd.DataFrame) -> float:
    hit_counts = sum((sim[c] >= 10).astype(int) for c in _DOUBLE_DOUBLE_COLUMNS)
    return float((hit_counts >= 2).mean())


def prob_triple_double(sim: pd.DataFrame) -> float:
    hit_counts = sum((sim[c] >= 10).astype(int) for c in _DOUBLE_DOUBLE_COLUMNS)
    return float((hit_counts >= 3).mean())


def prob_stat_and_team_win(sim: pd.DataFrame, stat: str, line: float, team_win_prob: float) -> float:
    """Player hits a stat line AND their team wins. Treats the two as
    independent (team outcome is a many-player aggregate, one player's
    individual box-score variance has only a marginal effect on it), which
    is a reasonable simplification -- but this stays a simplification, not
    a validated correlation estimate, and should be revisited if backtest
    data on team-outcome-conditional player stats becomes available.
    """
    return prob_over(sim, stat, line) * team_win_prob
