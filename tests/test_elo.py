import pandas as pd

from wcwinner.features.elo import compute_elo_history, expected_score, goal_diff_weight, tournament_k_factor


def test_expected_score_symmetric_at_zero_diff():
    assert expected_score(0.0) == 0.5


def test_expected_score_favors_higher_rating():
    assert expected_score(400) > 0.9
    assert expected_score(-400) < 0.1


def test_goal_diff_weight_matches_dixon_coles_elo_convention():
    assert goal_diff_weight(0) == 1.0
    assert goal_diff_weight(1) == 1.0
    assert goal_diff_weight(2) == 1.5
    assert goal_diff_weight(3) == (11 + 3) / 8


def test_tournament_k_factor_tiers():
    assert tournament_k_factor("Friendly") == 20
    assert tournament_k_factor("FIFA World Cup qualification") == 40
    assert tournament_k_factor("FIFA World Cup") == 60
    assert tournament_k_factor("Some Regional Cup Nobody Has Heard Of") == 30


def test_elo_update_direction_and_symmetry():
    results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "home_score": [2],
            "away_score": [0],
            "tournament": ["Friendly"],
            "neutral": [True],
        }
    )
    _, ratings = compute_elo_history(results)
    assert ratings["A"] > 1500
    assert ratings["B"] < 1500
    # Zero-sum on a neutral pitch: rating gained by the winner equals rating lost by the loser.
    assert abs((ratings["A"] - 1500) + (ratings["B"] - 1500)) < 1e-9


def test_unplayed_fixture_does_not_update_ratings():
    results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "home_score": [float("nan")],
            "away_score": [float("nan")],
            "tournament": ["Friendly"],
            "neutral": [True],
        }
    )
    _, ratings = compute_elo_history(results)
    assert ratings["A"] == 1500
    assert ratings["B"] == 1500
