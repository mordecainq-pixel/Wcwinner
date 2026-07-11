"""Basketball's answer to Dixon-Coles: a fitted offense/defense rating model
predicting point margin and total directly, rather than modeling "goals" as
Poisson events the way soccer does. WNBA teams score 70-90+ points a game;
that's a large-count, roughly continuous quantity well-approximated by a
Normal distribution, not a rare discrete-event process.

Structurally this mirrors the Dixon-Coles fit closely on purpose: each team
gets an attack and defense rating (here, additive point contributions rather
than Dixon-Coles' log-linear multipliers), a global home-advantage
parameter, and a fitted residual spread (sigma) standing in for Dixon-
Coles's rho correlation term. Same time-decay weighting, same L2
regularization for stability, same Elo-blended empirical-Bayes shrinkage
for small-sample teams as the soccer model -- only the likelihood family
changed to match the sport.

Predicting margin and total directly (rather than separate home/away score
distributions) is deliberate: those are exactly the two real betting
markets (spread and total), so this model's outputs plug straight into the
same Bet Builder architecture used for soccer.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from wnba.config import (
    CALIBRATION_A,
    CALIBRATION_B,
    DATA_PROCESSED_DIR,
    MARGIN_L2_REG,
    MARGIN_LOOKBACK_YEARS,
    MARGIN_MIN_GAMES_FOR_ELO_REGRESSION,
    MARGIN_SHRINKAGE_K,
    MARGIN_XI_TIME_DECAY,
)

MODEL_PATH = DATA_PROCESSED_DIR / "margin_model.pkl"


def _apply_calibration(p: float, a: float = CALIBRATION_A, b: float = CALIBRATION_B) -> float:
    """Platt scaling: pulls overconfident raw probabilities back toward 50%
    (a<1) based on walk-forward validated evidence the raw model runs hot at
    the high-confidence end - see config.py for the validation writeup.
    """
    p = np.clip(p, 1e-9, 1 - 1e-9)
    raw_logit = np.log(p / (1 - p))
    calibrated_logit = a * raw_logit + b
    return float(1.0 / (1.0 + np.exp(-calibrated_logit)))


@dataclass
class MarginModel:
    attack: dict[str, float]  # points scored above/below a league-average offense
    defense: dict[str, float]  # points allowed above/below a league-average defense (higher = worse)
    home_advantage: float
    sigma: float  # residual std dev of a single team's score around its predicted value
    league_avg_points: float
    game_counts: dict[str, int]
    raw_attack: dict[str, float] = field(default_factory=dict)
    raw_defense: dict[str, float] = field(default_factory=dict)
    elo_regression: dict[str, float] = field(default_factory=dict)

    def predicted_scores(self, home_team: str, away_team: str, neutral: bool = True) -> tuple[float, float]:
        a_home = self.attack.get(home_team, 0.0)
        d_home = self.defense.get(home_team, 0.0)
        a_away = self.attack.get(away_team, 0.0)
        d_away = self.defense.get(away_team, 0.0)
        adv = 0.0 if neutral else self.home_advantage
        pred_home = self.league_avg_points + a_home - d_away + adv
        pred_away = self.league_avg_points + a_away - d_home
        return pred_home, pred_away

    def margin_distribution(self, home_team: str, away_team: str, neutral: bool = True) -> tuple[float, float]:
        """(mean, std) of home_score - away_score."""
        pred_home, pred_away = self.predicted_scores(home_team, away_team, neutral)
        return pred_home - pred_away, self.sigma * np.sqrt(2)

    def total_distribution(self, home_team: str, away_team: str, neutral: bool = True) -> tuple[float, float]:
        """(mean, std) of home_score + away_score."""
        pred_home, pred_away = self.predicted_scores(home_team, away_team, neutral)
        return pred_home + pred_away, self.sigma * np.sqrt(2)

    def win_probability(self, home_team: str, away_team: str, neutral: bool = True) -> float:
        mu, sd = self.margin_distribution(home_team, away_team, neutral)
        raw = float(1.0 - norm.cdf(0.0, mu, sd))  # P(margin > 0); basketball has no draws
        return _apply_calibration(raw)

    def prob_over(self, home_team: str, away_team: str, line: float, neutral: bool = True) -> float:
        mu, sd = self.total_distribution(home_team, away_team, neutral)
        raw = float(1.0 - norm.cdf(line, mu, sd))
        return _apply_calibration(raw)

    def prob_home_covers_spread(self, home_team: str, away_team: str, point: float, neutral: bool = True) -> float:
        """P(home_margin + point > 0), e.g. point=-5.5 means home must win by 6+."""
        mu, sd = self.margin_distribution(home_team, away_team, neutral)
        raw = float(1.0 - norm.cdf(-point, mu, sd))
        return _apply_calibration(raw)


def _prepare_training_frame(results: pd.DataFrame, as_of: pd.Timestamp | None = None, xi: float = MARGIN_XI_TIME_DECAY) -> pd.DataFrame:
    df = results.copy()
    as_of = as_of or df["date"].max()
    df = df[df["date"] <= as_of]
    cutoff = as_of - pd.Timedelta(days=365 * MARGIN_LOOKBACK_YEARS)
    df = df[df["date"] >= cutoff]
    df["days_ago"] = (as_of - df["date"]).dt.days
    playoff_weight = np.where(df["is_playoff"], 1.3, 1.0)  # playoff results still matter more, like Elo's K-bump
    df["weight"] = np.exp(-xi * df["days_ago"]) * playoff_weight
    return df


def _neg_log_likelihood(params: np.ndarray, home_idx, away_idx, home_pts, away_pts, neutral, weights, n_teams: int, league_avg: float) -> float:
    attack = params[:n_teams]
    defense = params[n_teams : 2 * n_teams]
    home_adv = params[2 * n_teams]
    log_sigma = params[2 * n_teams + 1]
    sigma = np.exp(log_sigma)

    adv = np.where(neutral, 0.0, home_adv)
    pred_home = league_avg + attack[home_idx] - defense[away_idx] + adv
    pred_away = league_avg + attack[away_idx] - defense[home_idx]

    log_lik = norm.logpdf(home_pts, pred_home, sigma) + norm.logpdf(away_pts, pred_away, sigma)
    reg = MARGIN_L2_REG * (np.sum(attack**2) + np.sum(defense**2))
    return -np.sum(weights * log_lik) + reg


def _blend_with_elo(raw_attack, raw_defense, game_counts, elo_ratings) -> tuple[dict, dict, dict]:
    teams = list(raw_attack.keys())
    elo_arr = np.array([elo_ratings.get(t, 1500.0) for t in teams])
    attack_arr = np.array([raw_attack[t] for t in teams])
    defense_arr = np.array([raw_defense[t] for t in teams])
    counts_arr = np.array([game_counts.get(t, 0) for t in teams])

    well_estimated = counts_arr >= MARGIN_MIN_GAMES_FOR_ELO_REGRESSION
    if well_estimated.sum() < 5:
        return dict(raw_attack), dict(raw_defense), {}

    attack_slope, attack_intercept = np.polyfit(elo_arr[well_estimated], attack_arr[well_estimated], 1)
    defense_slope, defense_intercept = np.polyfit(elo_arr[well_estimated], defense_arr[well_estimated], 1)

    elo_implied_attack = attack_slope * elo_arr + attack_intercept
    elo_implied_defense = defense_slope * elo_arr + defense_intercept

    shrink_weight = counts_arr / (counts_arr + MARGIN_SHRINKAGE_K)
    blended_attack = shrink_weight * attack_arr + (1 - shrink_weight) * elo_implied_attack
    blended_defense = shrink_weight * defense_arr + (1 - shrink_weight) * elo_implied_defense

    regression = {
        "attack_slope": attack_slope, "attack_intercept": attack_intercept,
        "defense_slope": defense_slope, "defense_intercept": defense_intercept,
    }
    return dict(zip(teams, blended_attack)), dict(zip(teams, blended_defense)), regression


def fit(results: pd.DataFrame, elo_ratings: dict[str, float], as_of: pd.Timestamp | None = None, xi: float = MARGIN_XI_TIME_DECAY) -> MarginModel:
    df = _prepare_training_frame(results, as_of, xi)

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = df["home_team"].map(team_index).to_numpy()
    away_idx = df["away_team"].map(team_index).to_numpy()
    home_pts = df["home_score"].to_numpy(dtype=float)
    away_pts = df["away_score"].to_numpy(dtype=float)
    neutral = df["neutral"].to_numpy(dtype=bool)
    weights = df["weight"].to_numpy()
    league_avg = float(np.concatenate([home_pts, away_pts]).mean())

    game_counts = pd.concat([df["home_team"], df["away_team"]]).value_counts().to_dict()

    x0 = np.zeros(2 * n_teams + 2)
    x0[2 * n_teams] = 3.0  # home advantage warm start (points)
    x0[2 * n_teams + 1] = np.log(10.0)  # sigma warm start

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(home_idx, away_idx, home_pts, away_pts, neutral, weights, n_teams, league_avg),
        method="L-BFGS-B",
        bounds=[(-30, 30)] * n_teams + [(-30, 30)] * n_teams + [(-5, 20), (np.log(3), np.log(25))],
    )

    attack = result.x[:n_teams]
    defense = result.x[n_teams : 2 * n_teams]
    home_adv = float(result.x[2 * n_teams])
    sigma = float(np.exp(result.x[2 * n_teams + 1]))

    raw_attack = dict(zip(teams, attack))
    raw_defense = dict(zip(teams, defense))
    blended_attack, blended_defense, regression = _blend_with_elo(raw_attack, raw_defense, game_counts, elo_ratings)

    return MarginModel(
        attack=blended_attack,
        defense=blended_defense,
        home_advantage=home_adv,
        sigma=sigma,
        league_avg_points=league_avg,
        game_counts={t: int(game_counts.get(t, 0)) for t in teams},
        raw_attack=raw_attack,
        raw_defense=raw_defense,
        elo_regression=regression,
    )


def save(model: MarginModel, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load(path: Path = MODEL_PATH) -> MarginModel:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the WNBA refresh pipeline first.")
    with open(path, "rb") as f:
        return pickle.load(f)
