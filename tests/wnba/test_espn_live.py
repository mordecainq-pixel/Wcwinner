import pandas as pd

from wnba.data.espn_live import team_roster


def test_team_roster_uses_only_recent_games_and_orders_by_last_played():
    df = pd.DataFrame([
        {"player_id": "1", "player_name": "Old Player", "position": "G", "team": "Fever", "date": pd.Timestamp("2024-01-01")},
        {"player_id": "2", "player_name": "Current Player", "position": "F", "team": "Fever", "date": pd.Timestamp("2025-06-01")},
        {"player_id": "2", "player_name": "Current Player", "position": "F", "team": "Fever", "date": pd.Timestamp("2025-06-03")},
        {"player_id": "3", "player_name": "Traded Away", "position": "C", "team": "Fever", "date": pd.Timestamp("2023-01-01")},
    ])
    roster = team_roster("Fever", df, n_recent_games=2)

    names = set(roster["player_name"])
    assert "Current Player" in names
    assert "Traded Away" not in names  # not among the 2 most recent game dates
    assert roster.iloc[0]["player_name"] == "Current Player"


def test_team_roster_empty_for_unknown_team():
    df = pd.DataFrame(columns=["player_id", "player_name", "position", "team", "date"])
    roster = team_roster("Nonexistent", df)
    assert roster.empty
