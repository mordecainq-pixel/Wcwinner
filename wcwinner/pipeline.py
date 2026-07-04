"""The one function `scripts/refresh_data.py` and `wcwinner refresh` both
call: re-download historical data, recompute Elo, refit Dixon-Coles, persist
both. Kept here (rather than duplicated between the script and the CLI) since
both entry points need the exact same steps.
"""
from __future__ import annotations

import pickle
import time

from wcwinner.config import ELO_RATINGS_PATH
from wcwinner.data import github_ingest, kaggle_ingest
from wcwinner.features.elo import compute_elo_history
from wcwinner.model import dixon_coles


def run_full_refresh(verbose: bool = True, source: str = "github") -> dict:
    """`source="github"` (default) pulls straight from martj42/international_results
    on GitHub -- no auth, no kagglehub, and raw.githubusercontent.com is
    trusted by default in most sandboxed network policies. `source="kaggle"`
    uses the original kagglehub-based path instead.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    if source == "github":
        log("Refreshing historical dataset from GitHub (martj42/international_results)...")
        github_ingest.refresh()
    elif source == "kaggle":
        log("Refreshing historical dataset from Kaggle...")
        kaggle_ingest.refresh()
    else:
        raise ValueError(f"Unknown source {source!r}; use 'github' or 'kaggle'")

    freshness = kaggle_ingest.data_freshness()  # reads data/raw/, same schema regardless of source
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
