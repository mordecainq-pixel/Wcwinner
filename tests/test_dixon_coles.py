import numpy as np
import pandas as pd

from wcwinner.model.dixon_coles import DixonColesModel, _tau, _tau_vectorized, fit


def test_tau_is_identity_when_rho_zero():
    for x, y in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]:
        assert _tau(x, y, lam=1.3, mu=0.9, rho=0.0) == 1.0


def test_tau_matches_dixon_coles_1997_formula():
    lam, mu, rho = 1.2, 0.8, -0.1
    assert _tau(0, 0, lam, mu, rho) == 1 - lam * mu * rho
    assert _tau(0, 1, lam, mu, rho) == 1 + lam * rho
    assert _tau(1, 0, lam, mu, rho) == 1 + mu * rho
    assert _tau(1, 1, lam, mu, rho) == 1 - rho
    assert _tau(2, 2, lam, mu, rho) == 1.0


def test_tau_vectorized_matches_scalar():
    home_goals = np.array([0, 0, 1, 1, 2])
    away_goals = np.array([0, 1, 0, 1, 2])
    lam = np.full(5, 1.1)
    mu = np.full(5, 0.7)
    rho = -0.08
    vec = _tau_vectorized(home_goals, away_goals, lam, mu, rho)
    scalar = np.array([_tau(int(x), int(y), 1.1, 0.7, rho) for x, y in zip(home_goals, away_goals)])
    assert np.allclose(vec, scalar)


def test_score_matrix_sums_to_one():
    model = DixonColesModel(
        attack={"A": 0.3, "B": -0.1},
        defense={"A": -0.2, "B": 0.1},
        home_advantage=0.2,
        rho=-0.08,
        match_counts={"A": 50, "B": 50},
    )
    matrix = model.score_matrix("A", "B", neutral=True, max_goals=8)
    assert abs(matrix.sum() - 1.0) < 1e-9
    assert (matrix >= 0).all()


def test_expected_goals_home_advantage_only_applies_when_not_neutral():
    model = DixonColesModel(
        attack={"A": 0.0, "B": 0.0},
        defense={"A": 0.0, "B": 0.0},
        home_advantage=0.3,
        rho=0.0,
        match_counts={"A": 50, "B": 50},
    )
    lam_neutral, _ = model.expected_goals("A", "B", neutral=True)
    lam_home, _ = model.expected_goals("A", "B", neutral=False)
    assert lam_home > lam_neutral


def test_fit_recovers_stronger_attacking_team_from_synthetic_data():
    rng = np.random.default_rng(0)
    rows = []
    date = pd.Timestamp("2023-01-01")
    for _ in range(60):
        # "A" is a genuinely stronger scoring team than "B" by construction.
        hs = rng.poisson(2.0)
        as_ = rng.poisson(0.6)
        rows.append(["A", "B", hs, as_, date, "Friendly", True])
        date += pd.Timedelta(days=3)
    df = pd.DataFrame(rows, columns=["home_team", "away_team", "home_score", "away_score", "date", "tournament", "neutral"])

    model = fit(df, elo_ratings={"A": 1700, "B": 1400})
    assert model.attack["A"] > model.attack["B"]
