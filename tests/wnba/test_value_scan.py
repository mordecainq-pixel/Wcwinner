import pandas as pd
import pytest

from wnba import value_scan
from wnba.betbuilder import MatchOptions
from wnba.model.player_model import compute_opponent_factors


class _StubClient:
    def __init__(self, prop_response, milestone_response):
        self.prop_response = prop_response
        self.milestone_response = milestone_response

    def event_player_props(self, event_id, markets, regions=None, bookmakers=None):
        if "double_double" in markets or "triple_double" in markets:
            return self.milestone_response
        return self.prop_response


def _player_box():
    return pd.DataFrame({
        "player_id": ["p1"] * 5,
        "player_name": ["Home Star"] * 5,
        "team": ["Home"] * 5,
        "opponent": ["X"] * 5,
        "date": pd.date_range("2025-01-01", periods=5, freq="3D"),
        "points": [16, 18, 20, 14, 19],
        "rebounds": [4, 5, 3, 6, 4],
        "assists": [3, 2, 4, 3, 5],
        "fg3m": [2, 1, 3, 0, 2],
        "steals": [1, 0, 1, 2, 1],
        "blocks": [0, 0, 1, 0, 0],
        "turnovers": [2, 1, 2, 3, 1],
    })


def test_scan_bookmaker_value_computes_gap_and_ranks_by_ev():
    player_box = _player_box()
    opponent_factors = compute_opponent_factors(player_box)

    prop_response = {"bookmakers": [{"key": "fanduel", "markets": [{"key": "player_points", "outcomes": [
        {"name": "Over", "description": "Home Star", "price": 1.9, "point": 14.5},
        {"name": "Under", "description": "Home Star", "price": 1.9, "point": 14.5},
    ]}]}]}
    milestone_response = {"bookmakers": []}
    client = _StubClient(prop_response, milestone_response)

    match = MatchOptions("Home", "Away", "2026-07-05", legs=[], event_id="evt1")
    rows = value_scan.scan_bookmaker_value([match], player_box, opponent_factors, "fanduel", client=client)

    assert len(rows) == 1
    row = rows[0]
    assert row.player_name == "Home Star"
    assert row.stat == "points"
    assert row.book_line == 14.5
    assert row.model_mean == pytest.approx(row.gap + 14.5)
    assert row.best_side in ("over", "under")
    assert row.best_ev == max(row.ev_over, row.ev_under)


def test_scan_bookmaker_value_skips_matches_without_event_id():
    match = MatchOptions("Home", "Away", "2026-07-05", legs=[])
    rows = value_scan.scan_bookmaker_value([match], pd.DataFrame(), pd.DataFrame(), "fanduel", client=_StubClient({}, {}))
    assert rows == []


def test_scan_bookmaker_value_sorts_by_ev_descending():
    player_box = _player_box()
    opponent_factors = compute_opponent_factors(player_box)
    # two lines for the same player: one clearly good value (low line), one bad (high line)
    prop_response = {"bookmakers": [{"key": "fanduel", "markets": [
        {"key": "player_points", "outcomes": [
            {"name": "Over", "description": "Home Star", "price": 1.9, "point": 5.5},
            {"name": "Under", "description": "Home Star", "price": 1.9, "point": 5.5},
        ]},
        {"key": "player_rebounds", "outcomes": [
            {"name": "Over", "description": "Home Star", "price": 1.9, "point": 25.5},
            {"name": "Under", "description": "Home Star", "price": 1.9, "point": 25.5},
        ]},
    ]}]}
    client = _StubClient(prop_response, {"bookmakers": []})
    match = MatchOptions("Home", "Away", "2026-07-05", legs=[], event_id="evt1")
    rows = value_scan.scan_bookmaker_value([match], player_box, opponent_factors, "fanduel", client=client)

    assert len(rows) == 2
    assert rows[0].best_ev >= rows[1].best_ev


def test_save_and_load_scan_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(value_scan, "VALUE_SCAN_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(value_scan, "VALUE_SCAN_CACHE_FILE", tmp_path / "cache" / "value_scan.json")

    row = value_scan.ValueRow(
        home_team="Home", away_team="Away", date="2026-07-05", player_name="Home Star", stat="points",
        market_type="prop", book_line=14.5, model_mean=18.0, gap=3.5, model_prob_over=0.7,
        market_prob_over=0.5, over_odds=1.9, under_odds=1.9, ev_over=0.33, ev_under=-0.05,
        best_side="over", best_ev=0.33,
    )
    value_scan.save_scan([row], "fanduel")

    loaded = value_scan.load_cached_scan()
    assert loaded["bookmaker"] == "fanduel"
    assert len(loaded["rows"]) == 1
    assert loaded["rows"][0]["player_name"] == "Home Star"
    assert "fetched_at" in loaded


def test_load_cached_scan_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(value_scan, "VALUE_SCAN_CACHE_FILE", tmp_path / "nope.json")
    assert value_scan.load_cached_scan() is None
