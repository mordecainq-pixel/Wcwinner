import numpy as np
import pandas as pd

from wnba.model.player_model import compute_opponent_factors
from wnba.validate.player_backtest import compare_injury_adjustment, pit_values, summarize_pit


def _synthetic_player_box(n_games=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_games, freq="3D")
    opponents = rng.choice(["Team A", "Team B", "Team C"], size=n_games)
    points = rng.normal(18, 5, size=n_games).round().clip(min=0)
    return pd.DataFrame({
        "player_id": ["p1"] * n_games,
        "date": dates,
        "opponent": opponents,
        "team": ["Home"] * n_games,
        "points": points,
        "rebounds": rng.normal(6, 2, size=n_games).round().clip(min=0),
        "assists": rng.normal(4, 2, size=n_games).round().clip(min=0),
        "fg3m": rng.normal(2, 1, size=n_games).round().clip(min=0),
        "steals": rng.normal(1, 1, size=n_games).round().clip(min=0),
        "blocks": rng.normal(0.5, 0.7, size=n_games).round().clip(min=0),
        "turnovers": rng.normal(2, 1, size=n_games).round().clip(min=0),
    })


def test_pit_values_are_bounded_and_only_computed_with_enough_prior_history():
    df = _synthetic_player_box(n_games=40)
    factors = compute_opponent_factors(df)
    pit = pit_values(df, factors, cutoff_date="2024-03-01", n_sims=200, min_prior_games=5, rng=np.random.default_rng(1))

    assert not pit.empty
    for stat in ["points", "rebounds", "assists"]:
        assert pit[stat].between(0, 1).all()


def _injury_backtest_scenario():
    """10 independent one-off "shooter" players (so no test row ever becomes
    another test row's bootstrap history -- keeps the comparison isolated to
    the injury adjustment itself, not walk-forward data accumulation), each
    with the same real-variance prior history against "Rival" and a single
    held-out test game where Rival's rotation player star1 was actually out
    and the shooter scored above their own typical range (26, vs a
    prior-history mean of 20) -- exactly the situation the injury-aware
    adjustment exists for.
    """
    star1 = pd.DataFrame({
        "player_id": ["star1"] * 20, "team": ["Rival"] * 20, "opponent": ["X"] * 20,
        "event_id": [f"r{i}" for i in range(20)], "date": pd.date_range("2023-01-01", periods=20, freq="3D"),
        "minutes": [30.0] * 20, "points": [15.0] * 20, "rebounds": [5.0] * 20, "assists": [3.0] * 20,
        "fg3m": [1.0] * 20, "steals": [1.0] * 20, "blocks": [0.5] * 20, "turnovers": [2.0] * 20,
    })

    prior_points_pool = [14.0, 17.0, 20.0, 23.0, 26.0, 17.0, 20.0, 23.0, 14.0, 26.0]  # mean 20, real spread
    prior_frames, test_frames, avail_rows = [], [], []
    for i in range(10):
        pid = f"shooter{i}"
        prior_dates = pd.date_range("2024-01-01", periods=10, freq="3D")
        prior = pd.DataFrame({
            "player_id": [pid] * 10, "team": ["Other"] * 10, "opponent": ["Rival"] * 10,
            "event_id": [f"p{i}_{j}" for j in range(10)], "date": prior_dates, "minutes": [30.0] * 10,
            "points": prior_points_pool, "rebounds": [4.0] * 10, "assists": [3.0] * 10,
            "fg3m": [1.0] * 10, "steals": [1.0] * 10, "blocks": [0.3] * 10, "turnovers": [2.0] * 10,
        })
        test_date = pd.Timestamp("2024-06-01")
        test = pd.DataFrame({
            "player_id": [pid], "team": ["Other"], "opponent": ["Rival"],
            "event_id": [f"test{i}"], "date": [test_date], "minutes": [30.0],
            "points": [26.0], "rebounds": [4.0], "assists": [3.0],
            "fg3m": [1.0], "steals": [1.0], "blocks": [0.3], "turnovers": [2.0],
        })
        prior_frames.append(prior)
        test_frames.append(test)
        avail_rows += [{"event_id": eid, "date": d, "team": "Rival", "player_id": "star1", "played": True}
                        for eid, d in zip(prior["event_id"], prior["date"])]
        avail_rows.append({"event_id": f"test{i}", "date": test_date, "team": "Rival", "player_id": "star1", "played": False})

    other_out_dates = pd.date_range("2024-03-01", periods=15, freq="1D")
    other_out = pd.DataFrame({
        "player_id": [f"o{i}" for i in range(15)], "team": ["Other"] * 15, "opponent": ["Rival"] * 15,
        "event_id": [f"oout{i}" for i in range(15)], "date": other_out_dates, "minutes": [28.0] * 15,
        "points": [28.0] * 15, "rebounds": [4.0] * 15, "assists": [3.0] * 15,
        "fg3m": [1.0] * 15, "steals": [1.0] * 15, "blocks": [0.3] * 15, "turnovers": [2.0] * 15,
    })
    avail_rows += [{"event_id": eid, "date": d, "team": "Rival", "player_id": "star1", "played": False}
                   for eid, d in zip(other_out["event_id"], other_out["date"])]

    player_box = pd.concat([star1] + prior_frames + test_frames + [other_out], ignore_index=True)
    availability = pd.DataFrame(avail_rows)
    return player_box, availability


def test_compare_injury_adjustment_improves_calibration_on_real_absence_games():
    player_box, availability = _injury_backtest_scenario()
    opponent_factors = compute_opponent_factors(player_box)

    baseline_pit, injury_pit = compare_injury_adjustment(
        player_box, availability, opponent_factors, cutoff_date="2024-05-01", n_sims=3000, min_prior_games=5,
    )

    assert len(baseline_pit) == len(injury_pit) == 10
    assert baseline_pit["points"].between(0, 1).all()
    assert injury_pit["points"].between(0, 1).all()
    # baseline doesn't know star1 is out -- it should systematically undershoot
    # the actual (mean PIT pulled well above 0.5). Injury-aware, having bumped
    # the opponent factor up from real history of star1's missed games,
    # should land noticeably closer to the well-calibrated 0.5.
    assert baseline_pit["points"].mean() > 0.65
    assert injury_pit["points"].mean() < baseline_pit["points"].mean()
    assert abs(injury_pit["points"].mean() - 0.5) < abs(baseline_pit["points"].mean() - 0.5)


def test_summarize_pit_reports_mean_near_half_for_stable_stationary_process():
    df = _synthetic_player_box(n_games=80)
    factors = compute_opponent_factors(df)
    pit = pit_values(df, factors, cutoff_date="2024-02-01", n_sims=300, min_prior_games=5, rng=np.random.default_rng(2))
    summary = summarize_pit(pit, stat_cols=["points", "rebounds", "assists"])

    assert len(summary) == 3
    # a stationary (no real trend/role-change) synthetic process should calibrate
    # roughly symmetrically -- loose bound since this is a small sample
    assert summary["mean_pit"].between(0.3, 0.7).all()
