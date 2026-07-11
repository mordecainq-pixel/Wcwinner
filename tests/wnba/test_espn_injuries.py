from wnba.data.espn_injuries import team_injuries


def test_team_injuries_filters_to_requested_team():
    import pandas as pd
    df = pd.DataFrame([
        {"team": "Connecticut Sun", "player_name": "Saniya Rivers", "position": "G", "status": "Out", "date": "2026-07-11T03:37Z", "comment": "Ankle sprain."},
        {"team": "Atlanta Dream", "player_name": "Brionna Jones", "position": "F", "status": "Out", "date": "2026-07-11T18:06Z", "comment": "Knee."},
    ])
    result = team_injuries("Connecticut Sun", injuries=df)
    assert len(result) == 1
    assert result.iloc[0]["player_name"] == "Saniya Rivers"
