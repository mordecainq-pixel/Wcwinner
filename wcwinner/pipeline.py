"""The one function `scripts/refresh_data.py` and `wcwinner refresh` both
call: re-download historical data, recompute Elo, refit Dixon-Coles, persist
both. Kept here (rather than duplicated between the script and the CLI) since
both entry points need the exact same steps.
"""
from __future__ import annotations

import pickle
import time

from wcwinner.config import ELO_RATINGS_PATH
from wcwinner.data import kaggle_ingest
from wcwinner.features.elo import compute_elo_history
from wcwinner.model import dixon_coles


def run_full_refresh(verbose: bool = True) -> dict:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("Refreshing Kaggle historical dataset...")
    kaggle_ingest.refresh()
    freshness = kaggle_ingest.data_freshness()
    log(
        f"  {freshness['rows_played']} played matches, most recent completed "
        f"result: {freshness['max_played_date'].date()}"
    )

    log("Loading results and computing Elo ratings...")
    results = kaggle_ingest.load_results()
    _, elo_ratings = compute_elo_history(results)
    with open(ELO_RATINGS_PATH, "wb") as f:
        pickle.dump(elo_ratings, f)

    log("Fitting Dixon-Coles model (typically 15-30s)...")
    t0 = time.time()
    model = dixon_coles.fit(results, elo_ratings)
    dixon_coles.save(model)
    fit_seconds = time.time() - t0
    log(
        f"  fit in {fit_seconds:.1f}s, {len(model.attack)} teams, "
        f"home_advantage={model.home_advantage:.3f}, rho={model.rho:.3f}"
    )
    log(f"\nDone. Model -> {dixon_coles.MODEL_PATH}, Elo ratings -> {ELO_RATINGS_PATH}")

    return {"freshness": freshness, "model": model, "elo_ratings": elo_ratings, "fit_seconds": fit_seconds}
