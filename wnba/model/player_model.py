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
    PLAYER_INJURY_FACTOR_BOUNDS,
    PLAYER_INJURY_MIN_GAMES,
    PLAYER_INJURY_MIN_MINUTES,
    PLAYER_INJURY_SHRINKAGE_K,
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


def _qualifying_out_players(
    player_box: pd.DataFrame,
    team: str,
    out_player_ids: set[str],
    as_of: pd.Timestamp | None,
    min_minutes: float,
    min_games: int,
) -> list[str]:
    """Currently-out players worth conditioning on: enough of a track record
    on THIS team, at real rotation minutes. Filters out fringe/two-way
    scratches (e.g. a "coach's decision" DNP for someone who barely plays)
    so a single noisy bench absence can't move a projection -- only players
    whose own presence has a measurable footprint count.
    """
    history = player_box[player_box["team"] == team]
    if as_of is not None:
        history = history[history["date"] <= as_of]
    qualifying = []
    for pid in out_player_ids:
        rows = history[history["player_id"] == pid]
        if len(rows) >= min_games and rows["minutes"].mean() >= min_minutes:
            qualifying.append(pid)
    return qualifying


def compute_out_player_adjustment(
    player_box: pd.DataFrame,
    availability: pd.DataFrame,
    opponent_team: str,
    out_player_ids: set[str],
    stat_cols: list[str] = STAT_COLUMNS,
    as_of: pd.Timestamp | None = None,
    min_minutes: float = PLAYER_INJURY_MIN_MINUTES,
    min_games: int = PLAYER_INJURY_MIN_GAMES,
    shrinkage_k: float = PLAYER_INJURY_SHRINKAGE_K,
    factor_bounds: tuple[float, float] = PLAYER_INJURY_FACTOR_BOUNDS,
) -> pd.Series:
    """Multiplicative adjustment to `opponent_team`'s stats-allowed factor
    for currently-out rotation players, read off REAL history of this same
    team's own games without each player -- not an assumed point value.

    For each qualifying out player, compares what opponents actually scored
    against `opponent_team` in that player's historically missed games vs.
    all of `opponent_team`'s games, shrinking toward 1.0 (no effect) when
    there isn't much such history yet. Multiple simultaneous absences are
    combined by multiplying their marginal effects, which treats them as
    independent -- a simplification (same spirit as prob_stat_and_team_win's
    documented independence assumption), not a validated joint estimate.
    The combined result is clipped to `factor_bounds` so a couple of noisy
    small-sample marginal estimates can't compound into an extreme swing.
    """
    identity = pd.Series(1.0, index=stat_cols)
    if not out_player_ids:
        return identity

    qualifying = _qualifying_out_players(player_box, opponent_team, out_player_ids, as_of, min_minutes, min_games)
    if not qualifying:
        return identity

    faced = player_box[player_box["opponent"] == opponent_team]
    avail = availability[availability["team"] == opponent_team]
    if as_of is not None:
        faced = faced[faced["date"] <= as_of]
        avail = avail[avail["date"] <= as_of]
    if faced.empty:
        return identity
    overall_mean = faced[stat_cols].mean()

    combined = pd.Series(1.0, index=stat_cols)
    for pid in qualifying:
        missed_events = avail.loc[(avail["player_id"] == pid) & ~avail["played"], "event_id"].unique()
        if len(missed_events) == 0:
            continue
        out_games = faced[faced["event_id"].isin(missed_events)]
        if out_games.empty:
            continue
        marginal_raw = out_games[stat_cols].mean() / overall_mean
        weight = len(missed_events) / (len(missed_events) + shrinkage_k)
        marginal_shrunk = 1.0 + weight * (marginal_raw - 1.0)
        combined = combined * marginal_shrunk

    lo, hi = factor_bounds
    return combined.clip(lower=lo, upper=hi)


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
    out_player_ids: set[str] | None = None,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Weighted-bootstrap `n_sims` simulated game lines for one player
    against one opponent. Returns a DataFrame with `stat_cols` plus derived
    combo columns (pra, pr, pa, ra, sb).

    `out_player_ids` + `availability` are both optional and default to no
    adjustment -- pass both to additionally condition the opponent factor on
    real, currently-out rotation players via compute_out_player_adjustment.
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

    if out_player_ids and availability is not None:
        factor_row = factor_row * compute_out_player_adjustment(
            player_box, availability, opponent_team, out_player_ids, stat_cols=stat_cols, as_of=as_of,
        )

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
