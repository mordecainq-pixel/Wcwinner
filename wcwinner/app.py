"""Streamlit app: pick two teams, see win/draw/loss %, expected goals,
projected scoreline, and a scoreline probability heatmap.

Run with: streamlit run wcwinner/app.py
Requires `python scripts/refresh_data.py` to have been run at least once.
"""
from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import streamlit as st

from wcwinner.betbuilder import combine_legs, gather_candidate_legs
from wcwinner.config import ELO_RATINGS_PATH
from wcwinner.model import dixon_coles
from wcwinner.simulate.match import predict_match
from wcwinner.visualize import render_parlay_card


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

    tab_predict, tab_betbuilder = st.tabs(["Match Predictor", "Bet Builder"])
    with tab_predict:
        render_match_predictor(model, elo_ratings)
    with tab_betbuilder:
        render_bet_builder(model, elo_ratings)


def render_match_predictor(model, elo_ratings) -> None:
    teams = sorted(model.attack.keys())
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home / Team A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    with col2:
        default_away = "Brazil" if "Brazil" in teams else teams[1]
        away = st.selectbox("Away / Team B", teams, index=teams.index(default_away))

    neutral = st.checkbox("Neutral venue", value=True)
    knockout = st.checkbox("Knockout match (show ET/penalty advancement odds)", value=False)

    with st.expander("Squad news (manual, optional - no reliable free API for this)"):
        st.caption("Not automated by design. Leave at 1.0 unless you know of a specific absence to account for.")
        home_strength = st.slider(f"{home} squad strength multiplier", 0.5, 1.0, 1.0, 0.05)
        away_strength = st.slider(f"{away} squad strength multiplier", 0.5, 1.0, 1.0, 0.05)

    if home == away:
        st.warning("Pick two different teams.")
        return

    pred = predict_match(
        model, home, away, neutral=neutral, knockout=knockout, elo_ratings=elo_ratings,
        home_strength_multiplier=home_strength, away_strength_multiplier=away_strength,
    )

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


def render_bet_builder(model, elo_ratings) -> None:
    st.caption(
        "Every available match's best pick, ranked by real expected value (model probability "
        "vs. the actual betting market). You choose which ones go in your parlay - nothing is "
        "auto-selected or forced to hit a target. **Not a guarantee - for analysis/entertainment only.**"
    )

    legs = gather_candidate_legs(model, elo_ratings)
    if not legs:
        st.warning("No upcoming matches found.")
        return

    options = []
    for leg in legs:
        ev_str = f"{leg.ev*100:+.1f}% EV" if leg.market_prob is not None else "no market data"
        options.append(
            f"[{leg.market_type}] {leg.pick_label} -- {leg.odds:.2f}x -- model {leg.model_prob*100:.0f}% -- "
            f"{ev_str} -- {leg.home_team} vs {leg.away_team} ({leg.date})"
        )
    leg_by_option = dict(zip(options, legs))

    selected_options = st.multiselect("Pick your legs (as many as you want)", options)
    stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=5.0)

    if st.button("Build my parlay", type="primary"):
        if not selected_options:
            st.warning("Pick at least one match first.")
            return

        selected = [leg_by_option[o] for o in selected_options]
        result = combine_legs(selected, stake=stake)

        st.subheader(f"{len(result.legs)}-Leg Parlay")
        for i, leg in enumerate(result.legs, start=1):
            edge_str = f"{leg.edge*100:+.1f}% edge, {leg.ev*100:+.1f}% EV" if leg.edge is not None else "no market data"
            st.write(f"**{i}. [{leg.market_type}] {leg.pick_label}** -- {leg.odds:.2f}x ({edge_str}) -- {leg.home_team} vs {leg.away_team}, {leg.date}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Combined odds", f"{result.combined_odds:.2f}x")
        m2.metric("Est. chance all legs hit", f"{result.combined_model_prob*100:.1f}%")
        m3.metric("Payout", f"${result.payout:.2f}", delta=f"profit ${result.profit:.2f}")

        fig = render_parlay_card(result)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
