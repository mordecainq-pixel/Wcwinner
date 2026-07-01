"""Dixon-Coles adjusted Poisson model.

Each team gets an attack and defense strength fit by maximum likelihood
against the full (time-decayed) match history, plus a global home-advantage
parameter and the low-score correlation correction (rho / tau) from Dixon &
Coles (1997) for the 0-0/1-0/0-1/1-1 cells where plain independent Poisson
under-fits real scorelines.

Small-sample teams (e.g. Cape Verde, Bosnia) get shrunk toward an Elo-implied
prior rather than trusted at face value on a handful of noisy matches — see
`_blend_with_elo`.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from wcwinner.config import (
    DATA_PROCESSED_DIR,
    DC_L2_REG,
    DC_LOOKBACK_YEARS,
    DC_MIN_MATCHES_FOR_ELO_REGRESSION,
    DC_SHRINKAGE_K,
    DC_XI_TIME_DECAY,
)

MODEL_PATH = DATA_PROCESSED_DIR / "dixon_coles_model.pkl"


@dataclass
class DixonColesModel:
    attack: dict[str, float]
    defense: dict[str, float]
    home_advantage: float
    rho: float
    match_counts: dict[str, int]
    raw_attack: dict[str, float] = field(default_factory=dict)
    raw_defense: dict[str, float] = field(default_factory=dict)
    elo_regression: dict[str, float] = field(default_factory=dict)

    def expected_goals(self, home_team: str, away_team: str, neutral: bool = True) -> tuple[float, float]:
        a_home = self.attack.get(home_team, 0.0)
        d_home = self.defense.get(home_team, 0.0)
        a_away = self.attack.get(away_team, 0.0)
        d_away = self.defense.get(away_team, 0.0)
        gamma = 0.0 if neutral else self.home_advantage
        lam_home = np.exp(a_home + d_away + gamma)
        lam_away = np.exp(a_away + d_home)
        return float(lam_home), float(lam_away)

    def score_matrix(self, home_team: str, away_team: str, neutral: bool = True, max_goals: int = 8) -> np.ndarray:
        """P(home scores i, away scores j) for i,j in [0, max_goals], Dixon-Coles adjusted."""
        lam, mu = self.expected_goals(home_team, away_team, neutral)
        i = np.arange(max_goals + 1)
        home_pmf = poisson.pmf(i, lam)
        away_pmf = poisson.pmf(i, mu)
        matrix = np.outer(home_pmf, away_pmf)

        for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            matrix[x, y] *= _tau(x, y, lam, mu, self.rho)

        matrix /= matrix.sum()  # renormalize after the tau tweak and finite truncation
        return matrix


def _tau(x: int, y: int, lam: np.ndarray | float, mu: np.ndarray | float, rho: float):
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return np.ones_like(lam) if isinstance(lam, np.ndarray) else 1.0


def _tau_vectorized(home_goals: np.ndarray, away_goals: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lam)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    return tau


def _prepare_training_frame(results: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    df = results.dropna(subset=["home_score", "away_score"]).copy()
    as_of = as_of or df["date"].max()
    df = df[df["date"] <= as_of]
    cutoff = as_of - pd.Timedelta(days=365 * DC_LOOKBACK_YEARS)
    df = df[df["date"] >= cutoff]
    df["days_ago"] = (as_of - df["date"]).dt.days
    df["weight"] = np.exp(-DC_XI_TIME_DECAY * df["days_ago"])
    return df


def _neg_log_likelihood(params: np.ndarray, home_idx, away_idx, home_goals, away_goals, neutral, weights, n_teams: int) -> float:
    attack = params[:n_teams]
    defense = params[n_teams : 2 * n_teams]
    gamma = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    home_adv = np.where(neutral, 0.0, gamma)
    lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
    mu = np.exp(attack[away_idx] + defense[home_idx])

    log_lik = poisson.logpmf(home_goals, lam) + poisson.logpmf(away_goals, mu)
    tau = _tau_vectorized(home_goals, away_goals, lam, mu, rho)
    log_lik += np.log(np.clip(tau, 1e-10, None))

    reg = DC_L2_REG * (np.sum(attack**2) + np.sum(defense**2))
    return -np.sum(weights * log_lik) + reg


def _blend_with_elo(
    raw_attack: dict[str, float],
    raw_defense: dict[str, float],
    match_counts: dict[str, int],
    elo_ratings: dict[str, float],
) -> tuple[dict, dict, dict]:
    teams = list(raw_attack.keys())
    elo_arr = np.array([elo_ratings.get(t, 1500.0) for t in teams])
    attack_arr = np.array([raw_attack[t] for t in teams])
    defense_arr = np.array([raw_defense[t] for t in teams])
    counts_arr = np.array([match_counts.get(t, 0) for t in teams])

    well_estimated = counts_arr >= DC_MIN_MATCHES_FOR_ELO_REGRESSION
    if well_estimated.sum() < 5:
        # Not enough well-estimated teams to fit a meaningful prior; skip blending.
        return dict(raw_attack), dict(raw_defense), {}

    attack_slope, attack_intercept = np.polyfit(elo_arr[well_estimated], attack_arr[well_estimated], 1)
    defense_slope, defense_intercept = np.polyfit(elo_arr[well_estimated], defense_arr[well_estimated], 1)

    elo_implied_attack = attack_slope * elo_arr + attack_intercept
    elo_implied_defense = defense_slope * elo_arr + defense_intercept

    shrink_weight = counts_arr / (counts_arr + DC_SHRINKAGE_K)
    blended_attack = shrink_weight * attack_arr + (1 - shrink_weight) * elo_implied_attack
    blended_defense = shrink_weight * defense_arr + (1 - shrink_weight) * elo_implied_defense

    attack_out = dict(zip(teams, blended_attack))
    defense_out = dict(zip(teams, blended_defense))
    regression = {
        "attack_slope": attack_slope,
        "attack_intercept": attack_intercept,
        "defense_slope": defense_slope,
        "defense_intercept": defense_intercept,
    }
    return attack_out, defense_out, regression


def fit(results: pd.DataFrame, elo_ratings: dict[str, float], as_of: pd.Timestamp | None = None) -> DixonColesModel:
    """Fit the Dixon-Coles model by MLE on time-decayed match history, then
    blend small-sample teams' attack/defense toward their Elo-implied values.
    """
    df = _prepare_training_frame(results, as_of)

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = df["home_team"].map(team_index).to_numpy()
    away_idx = df["away_team"].map(team_index).to_numpy()
    home_goals = df["home_score"].to_numpy(dtype=int)
    away_goals = df["away_score"].to_numpy(dtype=int)
    neutral = df["neutral"].to_numpy(dtype=bool)
    weights = df["weight"].to_numpy()

    match_counts = pd.concat([df["home_team"], df["away_team"]]).value_counts().to_dict()

    x0 = np.zeros(2 * n_teams + 2)
    x0[2 * n_teams] = 0.2  # home advantage warm start
    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(home_idx, away_idx, home_goals, away_goals, neutral, weights, n_teams),
        method="L-BFGS-B",
        bounds=[(-3, 3)] * n_teams + [(-3, 3)] * n_teams + [(-1, 1), (-0.3, 0.3)],
    )

    attack = result.x[:n_teams]
    defense = result.x[n_teams : 2 * n_teams]
    gamma = float(result.x[2 * n_teams])
    rho = float(result.x[2 * n_teams + 1])

    raw_attack = dict(zip(teams, attack))
    raw_defense = dict(zip(teams, defense))

    blended_attack, blended_defense, regression = _blend_with_elo(raw_attack, raw_defense, match_counts, elo_ratings)

    return DixonColesModel(
        attack=blended_attack,
        defense=blended_defense,
        home_advantage=gamma,
        rho=rho,
        match_counts={t: int(match_counts.get(t, 0)) for t in teams},
        raw_attack=raw_attack,
        raw_defense=raw_defense,
        elo_regression=regression,
    )


def save(model: DixonColesModel, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load(path: Path = MODEL_PATH) -> DixonColesModel:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/refresh_data.py` first.")
    with open(path, "rb") as f:
        return pickle.load(f)
