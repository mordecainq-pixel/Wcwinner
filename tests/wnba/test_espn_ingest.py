from wnba.data.espn_ingest import _parse_event


def _event(season_type, comp_type_abbr="STD", completed=True, home_score=80, away_score=75):
    return {
        "date": "2025-07-20T23:00Z",
        "season": {"year": 2025, "type": season_type, "slug": "x"},
        "competitions": [
            {
                "status": {"type": {"completed": completed}},
                "type": {"id": "1", "abbreviation": comp_type_abbr},
                "neutralSite": False,
                "competitors": [
                    {"homeAway": "home", "team": {"displayName": "Indiana Fever"}, "score": str(home_score)},
                    {"homeAway": "away", "team": {"displayName": "New York Liberty"}, "score": str(away_score)},
                ],
            }
        ],
    }


def test_parses_a_valid_regular_season_game():
    parsed = _parse_event(_event(season_type=2))
    assert parsed is not None
    assert parsed["home_team"] == "Indiana Fever"
    assert parsed["away_team"] == "New York Liberty"
    assert parsed["home_score"] == 80
    assert parsed["away_score"] == 75
    assert parsed["is_playoff"] is False


def test_parses_a_valid_playoff_game():
    parsed = _parse_event(_event(season_type=3))
    assert parsed is not None
    assert parsed["is_playoff"] is True


def test_excludes_allstar_games():
    # Regression test: All-Star games ("Team Collier vs Team Clark" style
    # made-up rosters) were silently polluting real team Elo ratings before
    # this filter existed.
    parsed = _parse_event(_event(season_type=2, comp_type_abbr="ALLSTAR"))
    assert parsed is None


def test_excludes_preseason_exhibition_games():
    # Regression test: preseason games include exhibitions against non-WNBA
    # national teams (Japan, Australia, Puerto Rico ahead of the Olympics)
    # that were silently polluting real team Elo ratings before this filter.
    parsed = _parse_event(_event(season_type=1))
    assert parsed is None


def test_excludes_incomplete_games():
    parsed = _parse_event(_event(season_type=2, completed=False))
    assert parsed is None
