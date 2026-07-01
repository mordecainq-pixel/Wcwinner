"""Assembles the full per-match feature table: Elo, rolling form, rest days,
head-to-head goal differential, venue flag, and (where known) group/knockout
stage. Used for validation diagnostics and CLI/app "why this prediction"
context — the probability engine itself is wcwinner.model.dixon_coles.
"""
from __future__ import annotations

import pandas as pd

from wcwinner.features.elo import compute_elo_history
from wcwinner.features.rolling_form import compute_rolling_form


def build_feature_table(results: pd.DataFrame) -> pd.DataFrame:
    with_elo, _ = compute_elo_history(results)
    with_form = compute_rolling_form(results)

    features = with_elo.copy()
    for col in [
        "home_attack_form",
        "home_defense_form",
        "away_attack_form",
        "away_defense_form",
        "home_rest_days",
        "away_rest_days",
        "h2h_goal_diff",
    ]:
        features[col] = with_form[col].to_numpy()

    features["stage"] = pd.NA  # filled in for WC 2026 rows via attach_wc2026_stage
    return features


_STAGE_IS_KNOCKOUT = {
    "GROUP_STAGE": False,
    "LAST_32": True,
    "LAST_16": True,
    "QUARTER_FINALS": True,
    "SEMI_FINALS": True,
    "THIRD_PLACE": True,
    "FINAL": True,
}


def attach_wc2026_stage(features: pd.DataFrame, bracket_matches: list[dict]) -> pd.DataFrame:
    """Fill in the `stage` / `is_knockout` columns for World Cup 2026 rows using
    football-data.org's bracket state. Historical rows outside the live bracket
    keep stage=NA since Kaggle's generic history doesn't carry round-level
    detail to reconstruct this honestly.
    """
    stage_rows = [
        {
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "date": pd.Timestamp(m["utc_date"]).tz_localize(None).normalize(),
            "stage": m["stage"],
        }
        for m in bracket_matches
        if m["home_team"] and m["away_team"]
    ]
    stage_df = pd.DataFrame(stage_rows).drop_duplicates(subset=["home_team", "away_team", "date"])

    out = features.copy()
    out["_date_norm"] = out["date"].dt.normalize()
    out = out.merge(
        stage_df,
        left_on=["home_team", "away_team", "_date_norm"],
        right_on=["home_team", "away_team", "date"],
        how="left",
        suffixes=("", "_bracket"),
    )
    out["stage"] = out["stage_bracket"].combine_first(out["stage"])
    out["is_knockout"] = out["stage"].map(_STAGE_IS_KNOCKOUT)
    out = out.drop(columns=["_date_norm", "date_bracket", "stage_bracket"])
    return out
