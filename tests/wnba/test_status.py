from datetime import datetime, timedelta, timezone

import wnba.status as status


def test_read_status_defaults_to_idle_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "REFRESH_STATUS_PATH", tmp_path / "refresh_status.json")
    result = status.read_status()
    assert result["state"] == "idle"


def test_mark_updating_then_idle_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "REFRESH_STATUS_PATH", tmp_path / "refresh_status.json")
    status.mark_updating(eta_minutes=20)
    updating = status.read_status()
    assert updating["state"] == "updating"
    assert updating["eta_minutes"] == 20
    assert updating["started_at"] is not None

    status.mark_idle()
    idle = status.read_status()
    assert idle["state"] == "idle"
    assert idle["finished_at"] is not None


def test_minutes_since_started_computes_real_elapsed_time(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "REFRESH_STATUS_PATH", tmp_path / "refresh_status.json")
    ten_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    elapsed = status.minutes_since_started({"started_at": ten_min_ago})
    assert 9.9 < elapsed < 10.1


def test_minutes_since_started_none_when_not_updating():
    assert status.minutes_since_started({"started_at": None}) is None
