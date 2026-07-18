from wnba.data.espn_injuries import _extract_player_id, team_injuries


def test_team_injuries_filters_to_requested_team():
    import pandas as pd
    df = pd.DataFrame([
        {"team": "Connecticut Sun", "player_name": "Saniya Rivers", "position": "G", "status": "Out", "date": "2026-07-11T03:37Z", "comment": "Ankle sprain."},
        {"team": "Atlanta Dream", "player_name": "Brionna Jones", "position": "F", "status": "Out", "date": "2026-07-11T18:06Z", "comment": "Knee."},
    ])
    result = team_injuries("Connecticut Sun", injuries=df)
    assert len(result) == 1
    assert result.iloc[0]["player_name"] == "Saniya Rivers"


def test_extract_player_id_parses_playercard_link():
    athlete = {
        "displayName": "Taina Mair",
        "links": [
            {"rel": ["playercard", "desktop", "athlete"], "href": "https://www.espn.com/wnba/player/_/id/5106222/taina-mair"},
            {"rel": ["stats", "desktop", "athlete"], "href": "https://www.espn.com/wnba/player/stats/_/id/5106222/taina-mair"},
        ],
    }
    assert _extract_player_id(athlete) == "5106222"


def test_extract_player_id_missing_link_returns_none():
    assert _extract_player_id({"displayName": "No Links", "links": []}) is None
