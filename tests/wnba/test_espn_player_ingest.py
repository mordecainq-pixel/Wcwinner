from wnba.data.espn_player_ingest import _parse_athlete_stats, _parse_availability, _parse_boxscore

_KEYS = [
    "minutes", "points", "fieldGoalsMade-fieldGoalsAttempted",
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
    "freeThrowsMade-freeThrowsAttempted", "rebounds", "assists", "turnovers",
    "steals", "blocks", "offensiveRebounds", "defensiveRebounds", "fouls", "plusMinus",
]


def _athlete(name, stats, starter=True, position="G", player_id="1"):
    return {
        "athlete": {"id": player_id, "displayName": name, "position": {"abbreviation": position}},
        "starter": starter,
        "didNotPlay": not stats,
        "stats": stats,
    }


def _summary(home_athletes, away_athletes):
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"displayName": "Indiana Fever"},
                    "displayOrder": 1,
                    "statistics": [{"keys": _KEYS, "athletes": away_athletes}],
                },
                {
                    "team": {"displayName": "New York Liberty"},
                    "displayOrder": 0,
                    "statistics": [{"keys": _KEYS, "athletes": home_athletes}],
                },
            ]
        }
    }


def test_parses_a_full_stat_line():
    stats = ["35", "16", "4-10", "0-0", "8-8", "12", "4", "6", "2", "0", "3", "9", "3", "+17"]
    row = _parse_athlete_stats(_KEYS, stats)
    assert row["minutes"] == 35
    assert row["points"] == 16
    assert row["fgm"] == 4 and row["fga"] == 10
    assert row["fg3m"] == 0 and row["fg3a"] == 0
    assert row["ftm"] == 8 and row["fta"] == 8
    assert row["rebounds"] == 12
    assert row["assists"] == 4
    assert row["turnovers"] == 6
    assert row["steals"] == 2
    assert row["blocks"] == 0
    assert row["oreb"] == 3 and row["dreb"] == 9
    assert row["fouls"] == 3
    assert row["plus_minus"] == 17


def test_dnp_player_returns_none():
    assert _parse_athlete_stats(_KEYS, []) is None


def test_parse_boxscore_orders_teams_by_display_order_and_tags_opponent():
    home_athletes = [_athlete("Home Star", ["30", "20", "8-15", "1-3", "3-4", "5", "3", "2", "1", "0", "1", "4", "2", "+5"])]
    away_athletes = [_athlete("Away Star", ["28", "14", "5-12", "0-2", "4-6", "6", "2", "1", "0", "1", "2", "4", "1", "-5"])]
    rows = _parse_boxscore(_summary(home_athletes, away_athletes), event_id="123", date="2025-07-01", season=2025, is_playoff=False)

    assert len(rows) == 2
    by_name = {r["player_name"]: r for r in rows}
    assert by_name["Home Star"]["team"] == "New York Liberty"
    assert by_name["Home Star"]["opponent"] == "Indiana Fever"
    assert by_name["Away Star"]["team"] == "Indiana Fever"
    assert by_name["Away Star"]["opponent"] == "New York Liberty"
    assert by_name["Home Star"]["points"] == 20


def test_parse_boxscore_skips_dnp_players():
    home_athletes = [
        _athlete("Played", ["30", "20", "8-15", "1-3", "3-4", "5", "3", "2", "1", "0", "1", "4", "2", "+5"]),
        _athlete("Inactive", [], starter=False),
    ]
    rows = _parse_boxscore(_summary(home_athletes, []), event_id="123", date="2025-07-01", season=2025, is_playoff=False)
    names = {r["player_name"] for r in rows}
    assert "Played" in names
    assert "Inactive" not in names


def test_parse_availability_includes_dnp_players_with_played_false():
    home_athletes = [
        _athlete("Played", ["30", "20", "8-15", "1-3", "3-4", "5", "3", "2", "1", "0", "1", "4", "2", "+5"], player_id="10"),
        _athlete("Inactive", [], starter=False, player_id="11"),
    ]
    rows = _parse_availability(_summary(home_athletes, []), event_id="123", date="2025-07-01", season=2025, is_playoff=False)
    by_name = {r["player_name"]: r for r in rows}

    assert by_name["Played"]["played"] is True
    assert by_name["Inactive"]["played"] is False
    assert by_name["Played"]["team"] == "New York Liberty"
    assert by_name["Played"]["opponent"] == "Indiana Fever"
    assert by_name["Inactive"]["player_id"] == "11"
