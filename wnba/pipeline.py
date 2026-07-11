"""The one function `wnba refresh` calls: re-download team results, recompute
Elo, refit the margin model, and (optionally, since it's slow -- one HTTP
request per game) re-pull player box scores and opponent factors.
"""
from __future__ import annotations

import pickle
import time

from wnba.config import ELO_RATINGS_PATH, DATA_PROCESSED_DIR
from wnba.data import espn_ingest, espn_player_ingest
from wnba.features.elo import compute_elo_history
from wnba.model import margin_model
from wnba.model.player_model import compute_opponent_factors

OPPONENT_FACTORS_PATH = DATA_PROCESSED_DIR / "opponent_factors.pkl"


def run_full_refresh(verbose: bool = True, refresh_players: bool = False) -> dict:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("Refreshing team results from ESPN...")
    espn_ingest.refresh()
    results = espn_ingest.load_results()
    freshness = espn_ingest.data_freshness()
    log(f"  {freshness['rows_total']} games, most recent: {freshness['max_date'].date()}")

    log("Computing Elo ratings...")
    _, elo_ratings = compute_elo_history(results)
    with open(ELO_RATINGS_PATH, "wb") as f:
        pickle.dump(elo_ratings, f)

    log("Fitting margin/total model...")
    t0 = time.time()
    model = margin_model.fit(results, elo_ratings)
    margin_model.save(model)
    log(f"  fit in {time.time() - t0:.1f}s, {len(model.attack)} teams, home_advantage={model.home_advantage:.2f} pts")

    player_box = None
    if refresh_players:
        log("Refreshing player box scores from ESPN (one request per game, slow)...")
        t0 = time.time()
        player_box = espn_player_ingest.refresh()
        log(f"  {len(player_box)} player-game rows in {time.time() - t0:.0f}s")

        log("Computing opponent stat-allowed factors...")
        factors = compute_opponent_factors(player_box)
        OPPONENT_FACTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OPPONENT_FACTORS_PATH, "wb") as f:
            pickle.dump(factors, f)

    log("\nDone.")
    return {"freshness": freshness, "model": model, "elo_ratings": elo_ratings, "player_box": player_box}


def load_opponent_factors():
    if not OPPONENT_FACTORS_PATH.exists():
        raise FileNotFoundError(f"{OPPONENT_FACTORS_PATH} not found. Run refresh with refresh_players=True first.")
    with open(OPPONENT_FACTORS_PATH, "rb") as f:
        return pickle.load(f)
