import pandas as pd

from wnba.data.odds_client import _average_by_most_common_point, player_milestone_odds, player_prop_lines


class _StubClient:
    def __init__(self, response):
        self._response = response

    def event_player_props(self, event_id, markets, regions=None, bookmakers=None):
        return self._response


def test_average_by_most_common_point_ignores_outlier_lines():
    entries = [(17.5, 1.9), (17.5, 1.95), (20.5, 1.5)]  # 20.5 is a stray outlier line
    point, price = _average_by_most_common_point(entries)
    assert point == 17.5
    assert abs(price - 1.925) < 1e-9


def test_average_by_most_common_point_empty_returns_none():
    assert _average_by_most_common_point([]) is None


def test_player_prop_lines_averages_across_books_and_picks_consensus_line():
    response = {
        "bookmakers": [
            {"key": "fanduel", "markets": [{"key": "player_points", "outcomes": [
                {"name": "Over", "description": "Kelsey Mitchell", "price": 1.9, "point": 17.5},
                {"name": "Under", "description": "Kelsey Mitchell", "price": 1.9, "point": 17.5},
            ]}]},
            {"key": "draftkings", "markets": [{"key": "player_points", "outcomes": [
                {"name": "Over", "description": "Kelsey Mitchell", "price": 1.87, "point": 17.5},
                {"name": "Under", "description": "Kelsey Mitchell", "price": 1.95, "point": 17.5},
            ]}]},
        ]
    }
    df = player_prop_lines("evt1", stats=["points"], client=_StubClient(response))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["player_name"] == "Kelsey Mitchell"
    assert row["stat"] == "points"
    assert row["line"] == 17.5
    assert abs(row["market_over_prob"] + row["market_under_prob"] - 1.0) < 1e-9


def test_player_prop_lines_skips_one_sided_markets():
    response = {
        "bookmakers": [
            {"key": "fanduel", "markets": [{"key": "player_points", "outcomes": [
                {"name": "Over", "description": "Bench Player", "price": 1.9, "point": 5.5},
            ]}]},
        ]
    }
    df = player_prop_lines("evt1", stats=["points"], client=_StubClient(response))
    assert df.empty


def test_player_milestone_odds_averages_yes_prices_across_books():
    response = {
        "bookmakers": [
            {"key": "fanduel", "markets": [{"key": "player_double_double", "outcomes": [
                {"name": "Yes", "description": "A'ja Wilson", "price": 1.83},
            ]}]},
            {"key": "draftkings", "markets": [{"key": "player_double_double", "outcomes": [
                {"name": "Yes", "description": "A'ja Wilson", "price": 1.77},
            ]}]},
        ]
    }
    df = player_milestone_odds("evt1", client=_StubClient(response))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["player_name"] == "A'ja Wilson"
    assert row["stat"] == "double_double"
    assert abs(row["yes_odds"] - 1.80) < 1e-9
