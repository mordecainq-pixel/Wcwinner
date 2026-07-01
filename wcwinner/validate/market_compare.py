"""Compares the model's implied probabilities against The Odds API's
de-vigged market consensus for upcoming WC 2026 fixtures. Strictly a
calibration/edge check — market data never feeds the model itself.
"""
from __future__ import annotations

import pandas as pd

from wcwinner.data.odds_client import market_probabilities
from wcwinner.model.dixon_coles import DixonColesModel
from wcwinner.simulate.match import predict_match


def compare_to_market(model: DixonColesModel, market_df: pd.DataFrame | None = None) -> pd.DataFrame:
    market_df = market_df if market_df is not None else market_probabilities()

    rows = []
    for r in market_df.itertuples(index=False):
        pred = predict_match(model, r.home_team, r.away_team, neutral=True)
        rows.append(
            {
                "home_team": r.home_team,
                "away_team": r.away_team,
                "model_home_prob": pred["p_home_win"],
                "market_home_prob": r.market_home_prob,
                "home_edge": pred["p_home_win"] - r.market_home_prob,
                "model_draw_prob": pred["p_draw"],
                "market_draw_prob": r.market_draw_prob,
                "draw_edge": pred["p_draw"] - r.market_draw_prob,
                "model_away_prob": pred["p_away_win"],
                "market_away_prob": r.market_away_prob,
                "away_edge": pred["p_away_win"] - r.market_away_prob,
            }
        )
    return pd.DataFrame(rows)
