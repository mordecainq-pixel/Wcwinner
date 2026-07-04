"""Historical match data ingestion directly from martj42/international_results
on GitHub -- the actual upstream source the Kaggle dataset mirrors.

Preferred over kaggle_ingest.py where possible: raw.githubusercontent.com is
trusted by default in most sandboxed network policies (no "Full access"
toggle needed), needs no auth/token, and sidesteps kagglehub's fragile
dependency chain entirely (its kagglesdk dependency has shipped broken
wheels before - see kaggle_ingest.py's git history). Output files land in
the same data/raw/ paths with the same schema, so every downstream reader
(load_results, etc.) works unchanged regardless of which source populated
them.
"""
from __future__ import annotations

from pathlib import Path

import requests

from wcwinner.config import DATA_RAW_DIR

_BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
_FILES = ["results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv"]


def refresh() -> dict[str, Path]:
    """Download the latest CSVs straight from GitHub into data/raw/. Safe to
    run repeatedly (e.g. from cron) -- always fetches the current file, no
    local caching/versioning to go stale.
    """
    paths = {}
    for fname in _FILES:
        resp = requests.get(f"{_BASE_URL}/{fname}", timeout=30)
        resp.raise_for_status()
        dst = DATA_RAW_DIR / fname
        dst.write_bytes(resp.content)
        paths[fname] = dst
    return paths
