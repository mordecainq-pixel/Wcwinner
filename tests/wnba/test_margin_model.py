import numpy as np
import pandas as pd

from wnba.config import CALIBRATION_A, CALIBRATION_B
from wnba.model.margin_model import MarginModel, fit


def test_margin_and_total_distributions_are_consistent():
    model = MarginModel(
        attack={"A": 3.0, "B": -1.0},
        defense={"A": -2.0, "B": 1.0},
        home_advantage=2.5,
        sigma=10.0,
        league_avg_points=82.0,
        game_counts={"A": 100, "B": 100},
    )
    pred_home, pred_away = model.predicted_scores("A", "B", neutral=True)
    mu_margin, sd_margin = model.margin_distribution("A", "B", neutral=True)
    mu_total, sd_total = model.total_distribution("A", "B", neutral=True)

    assert abs(mu_margin - (pred_home - pred_away)) < 1e-9
    assert abs(mu_total - (pred_home + pred_away)) < 1e-9
    assert abs(sd_margin - sd_total) < 1e-9  # both are sqrt(2)*sigma here


def test_home_advantage_only_applies_when_not_neutral():
    model = MarginModel(
        attack={"A": 0.0, "B": 0.0}, defense={"A": 0.0, "B": 0.0},
        home_advantage=3.0, sigma=10.0, league_avg_points=82.0,
        game_counts={"A": 100, "B": 100},
    )
    neutral_margin, _ = model.margin_distribution("A", "B", neutral=True)
    home_margin, _ = model.margin_distribution("A", "B", neutral=False)
    assert home_margin > neutral_margin
    assert abs(home_margin - neutral_margin - 3.0) < 1e-9


def test_raw_margin_is_even_for_even_matchup_at_neutral_site():
    """The uncalibrated margin distribution -- the model's actual scoring
    logic -- must still be perfectly symmetric for a truly even matchup.
    """
    model = MarginModel(
        attack={"A": 0.0, "B": 0.0}, defense={"A": 0.0, "B": 0.0},
        home_advantage=3.0, sigma=10.0, league_avg_points=82.0,
        game_counts={"A": 100, "B": 100},
    )
    mu, _ = model.margin_distribution("A", "B", neutral=True)
    assert abs(mu) < 1e-9


def test_win_probability_for_even_matchup_matches_calibration_offset():
    """Platt scaling has a nonzero intercept (CALIBRATION_B), so a truly even
    matchup's calibrated win probability is NOT exactly 50% -- it's whatever
    the fitted calibration curve maps a raw-50% (logit=0) input to.
    """
    model = MarginModel(
        attack={"A": 0.0, "B": 0.0}, defense={"A": 0.0, "B": 0.0},
        home_advantage=3.0, sigma=10.0, league_avg_points=82.0,
        game_counts={"A": 100, "B": 100},
    )
    expected = 1.0 / (1.0 + np.exp(-(CALIBRATION_A * 0.0 + CALIBRATION_B)))
    assert abs(model.win_probability("A", "B", neutral=True) - expected) < 1e-9


def test_prob_home_covers_spread_matches_win_probability_at_zero_point():
    model = MarginModel(
        attack={"A": 3.0, "B": -1.0}, defense={"A": -2.0, "B": 1.0},
        home_advantage=2.5, sigma=10.0, league_avg_points=82.0,
        game_counts={"A": 100, "B": 100},
    )
    win_prob = model.win_probability("A", "B", neutral=True)
    cover_prob = model.prob_home_covers_spread("A", "B", point=0.0, neutral=True)
    assert abs(win_prob - cover_prob) < 1e-9


def test_fit_recovers_stronger_scoring_team_from_synthetic_data():
    rng = np.random.default_rng(0)
    rows = []
    d = pd.Timestamp("2024-05-01")
    for _ in range(60):
        home_pts = int(rng.normal(95, 8))  # "A" is genuinely a much better offense
        away_pts = int(rng.normal(75, 8))
        rows.append(["A", "B", home_pts, away_pts, d, False, False])
        d += pd.Timedelta(days=2)
    df = pd.DataFrame(rows, columns=["home_team", "away_team", "home_score", "away_score", "date", "is_playoff", "neutral"])

    model = fit(df, elo_ratings={"A": 1700, "B": 1400})
    assert model.attack["A"] > model.attack["B"]
    assert model.win_probability("A", "B", neutral=True) > 0.5
