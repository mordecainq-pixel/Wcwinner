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
# recency decay plus a hard lookback cutoff.
#
# Swept across MULTIPLE cutoffs (2022/2023/2024), not just one, after an
# initial single-cutoff sweep suggested going as short as a 7-day half-life
# (i.e. only last week's games really matter) - that turned out to be
# overfitting to that one test window: a 7-day half-life was actually one of
# the WORST choices at a different cutoff. A 20-day half-life scores nearly
# as well on average and is far more consistent across periods (never the
# worst at any tested cutoff). Confirms the real, sport-specific finding
# though: WNBA rewards a MUCH shorter memory than soccer's ~3-year one -
# rosters are 12-15 players and turn over hard year to year (trades, free
# agency), so recent form is a much stronger signal here than career-long
# history. Worth re-sweeping periodically as more seasons of data accumulate
# and the estimate gets less noisy.
MARGIN_XI_TIME_DECAY = 0.0347  # ~20-day half-life
MARGIN_LOOKBACK_YEARS = 8
MARGIN_L2_REG = 0.01
MARGIN_SHRINKAGE_K = 15  # empirical-Bayes shrinkage strength toward Elo-implied rating
MARGIN_MIN_GAMES_FOR_ELO_REGRESSION = 10

# --- Probability calibration (Platt scaling) ---
# The raw model was measurably overconfident at the high end (an ~84%-
# confidence bin resolved at ~63% observed - a real ~4.6 standard-error
# miss, not noise). Fit via walk-forward out-of-sample predictions across
# 4 non-overlapping cutoffs (2020/2021/2022/2023), each only scored on the
# following year so nothing in the calibration-fitting pool overlaps the
# final held-out test period. Confirmed the fix generalizes: evaluated on
# a genuinely separate final test period (2024-07 cutoff), log-loss
# improved 0.679 -> 0.669 and Brier 0.241 -> 0.238. calibrated_logit =
# CALIBRATION_A * raw_logit + CALIBRATION_B; CALIBRATION_A < 1 means "pull
# predictions back toward 50%," consistent with the overconfidence found.
CALIBRATION_A = 0.7682
CALIBRATION_B = -0.1341
