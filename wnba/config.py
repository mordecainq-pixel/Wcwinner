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

# Confirmed live against real upcoming games: game lines (h2h/spreads/totals)
# and most player-prop combos have real market data across FanDuel,
# DraftKings, ESPN BET, and others once both the "us" and "us2" regions are
# queried together (the default "us" region alone misses FanDuel/DraftKings
# player props on this API). Steals/blocks/turnovers/steals+blocks came back
# with no bookmaker offering them at all on the game checked -- kept in the
# market-key map anyway since coverage can vary by matchup, but expect them
# to often come back empty; the model projection still stands alone then.
#
# COST WARNING, learned the expensive way: The Odds API's free tier is 500
# credits/month, and per-EVENT player-prop requests are priced per
# market x region, not per request. Testing this integration (a handful of
# gather_match_options() calls across a 4-5 game slate, each pulling ~12
# player-prop markets x 2 regions x 2 endpoints (over/under + milestones)
# per event) burned the account from ~500 credits down to 2 in well under
# an hour. Game lines (game_market_probabilities, one bulk call for the
# whole slate) are cheap by comparison. Because of this, betbuilder.py
# deliberately does NOT fetch player props for every upcoming game up
# front -- only for games you've actually selected, same spirit as the
# existing "search scoped to chosen matches, not the whole schedule"
# design. Budget for real usage: at ~15-30 credits per game's worth of
# props, the free tier supports maybe 15-30 game-lookups a month, not a
# few hundred.
ODDS_API_REGIONS = "us,us2"
ODDS_API_PLAYER_PROP_MARKETS = {
    "points": "player_points",
    "rebounds": "player_rebounds",
    "assists": "player_assists",
    "fg3m": "player_threes",
    "steals": "player_steals",
    "blocks": "player_blocks",
    "turnovers": "player_turnovers",
    "pra": "player_points_rebounds_assists",
    "pr": "player_points_rebounds",
    "pa": "player_points_assists",
    "ra": "player_rebounds_assists",
    "sb": "player_blocks_steals",
}
ODDS_API_DOUBLE_DOUBLE_MARKET = "player_double_double"
ODDS_API_TRIPLE_DOUBLE_MARKET = "player_triple_double"

# League-wide injury report (site.web.api.espn.com, NOT the site.api.espn.com
# host used elsewhere -- confirmed live, no auth): real per-player status
# (Out/Questionable/Day-To-Day), injury type, and a human-written comment,
# refreshed continuously by ESPN. Unlike wcwinner's squad-news field, this
# doesn't need to be manual -- it's real, current, and free.
ESPN_INJURIES_URL = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries"

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

# --- Player props ---
# Player box scores come from ESPN's summary endpoint (one request per game,
# confirmed live back to at least 2015 -- see data/espn_player_ingest.py).
# That's far more requests than the one-per-season team backfill, so props
# use a shorter lookback: rosters/roles change constantly (trades, injuries,
# rotation changes), so a 3-year-old box score is weak signal for projecting
# a player's line next week anyway -- same "short memory" finding as the
# team model, just more extreme at the individual level.
PLAYER_LOOKBACK_YEARS = 3
PLAYER_BOX_MIN_REQUEST_INTERVAL = 1.0
PLAYER_FORM_HALFLIFE_GAMES = 10  # exponential recency weight, in games not days
PLAYER_MIN_GAMES_FOR_PROJECTION = 3
PLAYER_OPPONENT_SHRINKAGE_K = 30  # games faced before an opponent's "stats allowed" factor is trusted fully
PLAYER_SIM_COUNT = 10000

# --- Public deployment / admin panel ---
# The public site is meant to run unattended (GitHub Actions refreshes data
# daily and commits it back; Streamlit Community Cloud redeploys on every
# push). Admin access was IP-based at first, but confirmed live that
# Streamlit Community Cloud proxies requests through a pool of internal
# servers -- the detected "visitor IP" changed on every single reload, not
# just between sessions, so there was never a stable address to match
# against. Replaced with a hidden URL route (/adminlogs, see admin.py) +
# password instead, which doesn't depend on network/hosting quirks at all.
REFRESH_STATUS_PATH = DATA_PROCESSED_DIR / "refresh_status.json"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

# For the admin panel's "refresh now" button, which dispatches the same
# workflow the daily cron uses (see .github/workflows/wnba_daily_refresh.yml)
# via GitHub's API rather than running the (slow, ESPN-rate-limited) refresh
# inside the web app's own process.
ADMIN_GITHUB_TOKEN = os.getenv("ADMIN_GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "mordecainq-pixel/wcwinner")
GITHUB_WORKFLOW_FILE = "wnba_daily_refresh.yml"
GITHUB_WORKFLOW_REF = "claude/hello-fpti9r"  # keep in sync with the workflow's checkout ref above
