"""Central configuration: paths, credentials, and tunable model constants."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_RAW_DIR = PROJECT_ROOT / "data" / "wnba_raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "wnba_processed"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ELO_RATINGS_PATH = DATA_PROCESSED_DIR / "elo_ratings.pkl"

# ESPN's public (undocumented, no-auth) scoreboard API. Confirmed live and
# working for both current and historical dates back to at least 1998, and
# supports date-range queries (with an explicit `limit`, which otherwise
# silently truncates to 100) so a full season is one request, not ~180.
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
ESPN_HISTORICAL_LOOKBACK_YEARS = 15
ESPN_REQUEST_LIMIT = 1000

# Odds API — reuses the same .env key as wcwinner/, confirmed to cover WNBA.
import os  # noqa: E402

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_WNBA_SPORT_KEY = "basketball_wnba"

# --- Elo rating ---
# Basketball Elo conventionally uses a margin-of-victory multiplier rather
# than a fixed per-competition K-factor (there's no "World Cup vs qualifier"
# equivalent) - see features/elo.py for the exact formula, adapted from the
# publicly documented approach FiveThirtyEight used for their NBA Elo model.
ELO_BASE_K = 20
ELO_PLAYOFF_K_MULTIPLIER = 1.5  # playoff results count for more
ELO_INITIAL_RATING = 1500
ELO_HOME_ADVANTAGE = 70  # rating points; smaller relative effect than soccer's +100

# --- Margin/total model ---
# Same spirit as wcwinner's DC_XI_TIME_DECAY/DC_LOOKBACK_YEARS: exponential
# recency decay plus a hard lookback cutoff. NOT yet backtest-tuned the way
# the soccer model's decay rate was - these are reasonable starting priors,
# to be validated the same way before being trusted. See validate/backtest.py.
MARGIN_XI_TIME_DECAY = 0.0015  # ~462-day half-life, a starting prior pending its own sweep
MARGIN_LOOKBACK_YEARS = 8
MARGIN_L2_REG = 0.01
MARGIN_SHRINKAGE_K = 15  # empirical-Bayes shrinkage strength toward Elo-implied rating
MARGIN_MIN_GAMES_FOR_ELO_REGRESSION = 10
