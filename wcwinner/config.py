"""Central configuration: paths, credentials, and tunable model constants."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET_SLUG = "martj42/international-football-results-from-1872-to-2017"

# --- Credentials (all optional at import time; clients raise clearly if missing) ---
FOOTBALL_DATA_API_TOKEN = os.getenv("FOOTBALL_DATA_API_TOKEN", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_HOST = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_FOOTBALL_HOST = os.getenv("RAPIDAPI_FOOTBALL_HOST", "api-football-v1.p.rapidapi.com")

# --- Elo rating constants ---
# K-factor schedule follows the published World Football Elo Ratings methodology
# (eloratings.net/about): tournament importance scales the rating swing per match.
ELO_K_WORLD_CUP_FINALS = 60
ELO_K_CONTINENTAL_OR_INTERCONFEDERATION = 50
ELO_K_QUALIFIERS = 40
ELO_K_OTHER_TOURNAMENT = 30
ELO_K_FRIENDLY = 20
ELO_HOME_ADVANTAGE = 100  # rating points added to the home team's effective rating
ELO_INITIAL_RATING = 1500

# --- Rolling form (feature engineering) ---
ROLLING_FORM_WINDOW = 15  # max matches considered
ROLLING_FORM_HALF_LIFE = 6  # matches; exponential recency decay

# --- Dixon-Coles model ---
DC_XI_TIME_DECAY = 0.0018  # per-day exponential downweighting of older matches in MLE fit
DC_LOOKBACK_YEARS = 10  # matches older than this are excluded from fitting entirely;
# at DC_XI_TIME_DECAY's half-life (~385 days) they'd carry negligible weight anyway
DC_L2_REG = 0.001  # ridge penalty on attack/defense to keep the optimizer well-conditioned
DC_SHRINKAGE_K = 20  # empirical-Bayes shrinkage strength (in "equivalent matches") pulling
# low-sample teams' attack/defense toward their Elo-implied values
DC_MIN_MATCHES_FOR_ELO_REGRESSION = 15  # only well-estimated teams inform the Elo->attack/
# defense regression used to build that prior

# --- Knockout extra-time / penalty tie-break ---
# Regulation draws are resolved ~50/50 (extra time + penalties are close to a
# coin flip in reality) with a small tilt toward the higher-Elo team, capped
# via tanh so even huge Elo gaps can't push it far from 50/50.
KNOCKOUT_TIEBREAK_ELO_WEIGHT = 0.08
KNOCKOUT_TIEBREAK_ELO_SCALE = 400

# --- World Cup 2026 ---
WORLD_CUP_2026_START = "2026-06-11"
WORLD_CUP_2026_END = "2026-07-19"
FOOTBALL_DATA_WC_CODE = "WC"
API_FOOTBALL_WC_LEAGUE_ID = 1
ODDS_API_WC_SPORT_KEY = "soccer_fifa_world_cup"
