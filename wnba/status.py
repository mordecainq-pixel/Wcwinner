"""Refresh-status coordination between the GitHub Actions cron job and the
deployed Streamlit app -- they run on different machines with no shared
process state, so a small JSON file committed to the repo (and picked up on
the app's next redeploy/rerun) is the simplest reliable way to tell visitors
"data is updating, check back shortly" instead of a raw error or stale data
mid-write.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from wnba.config import REFRESH_STATUS_PATH

_DEFAULT_ETA_MINUTES = 25  # generous estimate for a full team+player refresh


def read_status() -> dict:
    if not REFRESH_STATUS_PATH.exists():
        return {"state": "idle", "started_at": None, "finished_at": None, "eta_minutes": None}
    with open(REFRESH_STATUS_PATH) as f:
        return json.load(f)


def mark_updating(eta_minutes: int = _DEFAULT_ETA_MINUTES) -> None:
    REFRESH_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REFRESH_STATUS_PATH, "w") as f:
        json.dump({
            "state": "updating",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "eta_minutes": eta_minutes,
        }, f)


def mark_idle() -> None:
    REFRESH_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REFRESH_STATUS_PATH, "w") as f:
        json.dump({
            "state": "idle",
            "started_at": None,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "eta_minutes": None,
        }, f)


def minutes_since_started(status: dict) -> float | None:
    if not status.get("started_at"):
        return None
    started = datetime.fromisoformat(status["started_at"])
    return (datetime.now(timezone.utc) - started).total_seconds() / 60.0
