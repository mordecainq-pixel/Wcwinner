"""Streamlit app: pick two teams, see win/draw/loss %, expected goals,
projected scoreline, and a scoreline probability heatmap.

Run with: streamlit run wcwinner/app.py
Requires `python scripts/refresh_data.py` to have been run at least once.
"""
from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import streamlit as st

from wcwinner.betbuilder import KNOWN_BOOKMAKERS, combine_legs, find_best_combo, gather_match_options
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


@st.cache_data(ttl=90, show_spinner="Fetching odds and bracket state...")
def _cached_match_options(_model, _elo_ratings, bookmakers):
    """Streamlit reruns the whole script on every widget interaction (dropdown,
    multiselect, stake field, button click) - without caching, each one of
    those refetches bracket state + odds from scratch, which is exactly what
    blew through football-data.org's 10-req/min free tier in practice. `_model`/
    `_elo_ratings` are underscore-prefixed so Streamlit doesn't try to hash
    the (unhashable) model object; only `bookmakers` determines the cache key.
    """
    return gather_match_options(_model, _elo_ratings, bookmakers=bookmakers)


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
        "Pick which matches interest you, optionally give a target payout, and it searches "
        "only within those matches (every bet type, not just moneyline) for the best fit - "
        "never a forced longshot to hit a number. **Not a guarantee - for analysis/entertainment only.**"
    )

    book_choice = st.selectbox(
        "Sportsbook", ["Average across all"] + KNOWN_BOOKMAKERS,
        help="Kalshi and Polymarket are not available through this data source.",
    )
    bookmakers = None if book_choice == "Average across all" else book_choice

    matches = _cached_match_options(model, elo_ratings, bookmakers)
    if not matches:
        st.warning("No upcoming matches found.")
        return

    match_labels = [f"{m.home_team} vs {m.away_team} ({m.date})" for m in matches]
    match_by_label = dict(zip(match_labels, matches))
    chosen_labels = st.multiselect("Which matches do you want in your parlay?", match_labels)

    if chosen_labels:
        st.caption("Best pick per chosen match, for reference:")
        for label in chosen_labels:
            b = match_by_label[label].best_leg
            st.write(f"- **{label}**: {b.pick_label} {b.odds:.2f}x -- {b.reasoning}")

    use_target = st.checkbox("Target a specific payout", value=False)
    target = st.number_input("Target payout (e.g. 5 = 5x)", min_value=1.1, value=5.0, step=0.5) if use_target else None
    stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=5.0)

    if st.button("Build my parlay", type="primary"):
        if not chosen_labels:
            st.warning("Pick at least one match first.")
            return

        chosen_matches = [match_by_label[label] for label in chosen_labels]

        if target is not None:
            selected = find_best_combo(chosen_matches, target)
            actual = 1.0
            for leg in selected:
                actual *= leg.odds
            if abs(actual - target) / target > 0.15:
                st.info(f"Closest available from these matches is {actual:.2f}x -- can't get near {target:.1f}x without adding more matches or accepting a longshot leg.")
        else:
            selected = [m.best_leg for m in chosen_matches]

        result = combine_legs(selected, stake=stake)

        st.subheader(f"{len(result.legs)}-Leg Parlay")
        for i, leg in enumerate(result.legs, start=1):
            st.write(f"**{i}. [{leg.market_type}] {leg.pick_label}** -- {leg.odds:.2f}x -- {leg.reasoning} -- {leg.home_team} vs {leg.away_team}, {leg.date}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Combined odds", f"{result.combined_odds:.2f}x")
        m2.metric("Est. chance all legs hit", f"{result.combined_model_prob*100:.1f}%")
        m3.metric("Payout", f"${result.payout:.2f}", delta=f"profit ${result.profit:.2f}")

        fig = render_parlay_card(result)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
