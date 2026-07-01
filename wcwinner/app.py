"""Streamlit app: pick two teams, see win/draw/loss %, expected goals,
projected scoreline, and a scoreline probability heatmap.

Run with: streamlit run wcwinner/app.py
Requires `python scripts/refresh_data.py` to have been run at least once.
"""
from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import streamlit as st

from wcwinner.config import ELO_RATINGS_PATH
from wcwinner.model import dixon_coles
from wcwinner.simulate.match import predict_match


@st.cache_resource
def load_model_and_elo():
    model = dixon_coles.load()
    with open(ELO_RATINGS_PATH, "rb") as f:
        elo_ratings = pickle.load(f)
    return model, elo_ratings


def main() -> None:
    st.set_page_config(page_title="WC 2026 Predictor", layout="centered")
    st.title("World Cup 2026 Match Predictor")
    st.caption("Dixon-Coles adjusted Poisson model, self-calculated Elo, fit on 1872-2026 history.")

    try:
        model, elo_ratings = load_model_and_elo()
    except FileNotFoundError:
        st.error("No fitted model found. Run `python scripts/refresh_data.py` first.")
        return

    teams = sorted(model.attack.keys())
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home / Team A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    with col2:
        default_away = "Brazil" if "Brazil" in teams else teams[1]
        away = st.selectbox("Away / Team B", teams, index=teams.index(default_away))

    neutral = st.checkbox("Neutral venue", value=True)
    knockout = st.checkbox("Knockout match (show ET/penalty advancement odds)", value=False)

    if home == away:
        st.warning("Pick two different teams.")
        return

    pred = predict_match(model, home, away, neutral=neutral, knockout=knockout, elo_ratings=elo_ratings)

    st.subheader(f"{home} vs {away}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win", f"{pred['p_home_win']*100:.1f}%")
    c2.metric("Draw", f"{pred['p_draw']*100:.1f}%")
    c3.metric(f"{away} win", f"{pred['p_away_win']*100:.1f}%")

    st.write(
        f"**Expected goals:** {home} {pred['expected_goals_home']:.2f} - "
        f"{pred['expected_goals_away']:.2f} {away}"
    )
    i, j = pred["most_likely_score"]
    st.write(f"**Most likely scoreline:** {i}-{j} ({pred['most_likely_score_prob']*100:.1f}%)")
    st.write(f"**Both teams to score:** {pred['p_btts']*100:.1f}%")

    if knockout:
        ko = pred["knockout"]
        st.subheader("Knockout advancement")
        c1, c2 = st.columns(2)
        c1.metric(f"{home} advances", f"{ko['p_home_advances']*100:.1f}%")
        c2.metric(f"{away} advances", f"{ko['p_away_advances']*100:.1f}%")
        st.caption(
            f"Regulation W/D/L: {ko['p_home_regulation_win']*100:.1f}% / "
            f"{ko['p_draw_regulation']*100:.1f}% / {ko['p_away_regulation_win']*100:.1f}% "
            f"-- if drawn, ET/penalty tiebreak favors {home} {ko['p_home_tiebreak_win']*100:.1f}%"
        )

    st.subheader("Scoreline probability heatmap")
    matrix = pred["score_matrix"][:6, :6]
    fig, ax = plt.subplots()
    im = ax.imshow(matrix * 100, cmap="Blues")
    ax.set_xlabel(f"{away} goals")
    ax.set_ylabel(f"{home} goals")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    for r in range(6):
        for c in range(6):
            ax.text(c, r, f"{matrix[r, c]*100:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Probability (%)")
    st.pyplot(fig)

    st.caption(
        f"Model fit stats for these teams -- {home}: {model.match_counts.get(home, 0)} matches in "
        f"training window, {away}: {model.match_counts.get(away, 0)} matches. Elo: {home} "
        f"{elo_ratings.get(home, float('nan')):.0f}, {away} {elo_ratings.get(away, float('nan')):.0f}."
    )


if __name__ == "__main__":
    main()
