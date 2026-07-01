import numpy as np

from wcwinner.validate.metrics import brier_score, log_loss, outcome_index


def test_outcome_index():
    assert outcome_index(2, 0) == 0
    assert outcome_index(1, 1) == 1
    assert outcome_index(0, 3) == 2


def test_log_loss_perfect_prediction_is_near_zero():
    probs = np.array([[0.999, 0.0005, 0.0005]])
    true_idx = np.array([0])
    assert log_loss(probs, true_idx) < 0.01


def test_log_loss_uniform_guess_matches_analytic_value():
    probs = np.full((10, 3), 1 / 3)
    true_idx = np.zeros(10, dtype=int)
    assert abs(log_loss(probs, true_idx) - np.log(3)) < 1e-9


def test_brier_score_perfect_prediction_is_zero():
    probs = np.array([[1.0, 0.0, 0.0]])
    true_idx = np.array([0])
    assert brier_score(probs, true_idx) == 0.0


def test_brier_score_worst_case_is_two():
    probs = np.array([[0.0, 0.0, 1.0]])
    true_idx = np.array([0])
    assert abs(brier_score(probs, true_idx) - 2.0) < 1e-9
