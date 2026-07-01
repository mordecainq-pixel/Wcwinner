"""Historical match data ingestion from the Kaggle "international football
results" dataset. This is the base for Elo ratings and Dixon-Coles fitting.

Automation: fully scriptable via `refresh()` / `scripts/refresh_data.py`, no
manual steps. Despite the "1872-to-2017" slug, the dataset is actively
maintained and includes in-progress World Cup 2026 results.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from wcwinner.config import DATA_RAW_DIR, KAGGLE_DATASET_SLUG

_FILES = ["results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv"]


def refresh() -> dict[str, Path]:
    """Download the latest version of the Kaggle dataset and copy its CSVs into
    data/raw/. Safe to run repeatedly (e.g. from cron); kagglehub caches the
    download and only re-fetches when a new dataset version is published.
    """
    import kagglehub

    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    paths = {}
    for fname in _FILES:
        src = cache_path / fname
        if not src.exists():
            continue
        dst = DATA_RAW_DIR / fname
        shutil.copyfile(src, dst)
        paths[fname] = dst
    return paths


def _raw_path(fname: str) -> Path:
    path = DATA_RAW_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/refresh_data.py` (or "
            "wcwinner.data.kaggle_ingest.refresh()) first."
        )
    return path


def _former_name_map() -> dict[str, str]:
    df = pd.read_csv(_raw_path("former_names.csv"))
    return dict(zip(df["former"], df["current"]))


def load_results(apply_renames: bool = True) -> pd.DataFrame:
    """Load the full match results history, oldest first.

    Columns: date, home_team, away_team, home_score, away_score, tournament,
    city, country, neutral. Rows with NaN scores are fixtures that haven't
    been played yet (used for upcoming-match lookups, excluded from fitting).
    """
    df = pd.read_csv(_raw_path("results.csv"), parse_dates=["date"])
    if apply_renames:
        renames = _former_name_map()
        df["home_team"] = df["home_team"].replace(renames)
        df["away_team"] = df["away_team"].replace(renames)
    return df.sort_values("date").reset_index(drop=True)


def load_goalscorers() -> pd.DataFrame:
    return pd.read_csv(_raw_path("goalscorers.csv"), parse_dates=["date"])


def load_shootouts() -> pd.DataFrame:
    return pd.read_csv(_raw_path("shootouts.csv"), parse_dates=["date"])


def played_matches(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Subset of `load_results()` restricted to matches with a final score."""
    df = df if df is not None else load_results()
    return df.dropna(subset=["home_score", "away_score"]).copy()


def data_freshness() -> dict:
    """Report the max date with a completed score vs. the max scheduled date,
    so a refresh run can log how current the data actually is.
    """
    df = load_results()
    played = played_matches(df)
    return {
        "rows_total": len(df),
        "rows_played": len(played),
        "max_played_date": played["date"].max(),
        "max_scheduled_date": df["date"].max(),
        "unplayed_fixture_rows": int(df["home_score"].isna().sum()),
    }
