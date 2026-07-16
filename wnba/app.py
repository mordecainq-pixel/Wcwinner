"""Streamlit app: matchup predictions, today's slate, player prop
projections, the live injury report, and a Bet Builder -- the WNBA
counterpart to wcwinner/app.py, same visual system.

Run with: streamlit run wnba/app.py
Requires `wnba-predictor refresh` to have been run at least once; player
props additionally require `wnba-predictor refresh --players`.

COST NOTE: The Odds API's free tier is 500 credits/month, and per-event
player-prop requests are priced per market x region -- expensive enough
that a handful of full-slate lookups can burn most of a month's quota (see
config.py's ODDS_API_REGIONS comment for the real numbers). Every tab that
touches live odds is gated behind an explicit button click and long cache
TTLs, never fetched automatically on page load or every widget rerun.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from wnba.betbuilder import KNOWN_BOOKMAKERS, add_player_prop_legs, combine_legs, find_best_combo, gather_match_options
from wnba.data.espn_injuries import fetch_injuries
from wnba.data.espn_live import team_roster, todays_games
from wnba.data.espn_player_ingest import load_player_boxscores
from wnba.data.odds_client import game_market_probabilities
from wnba.model import margin_model
from wnba.model.player_model import STAT_COLUMNS, STAT_LABELS, prob_double_double, prob_over, prob_triple_double, simulate_player_games
from wnba.pipeline import load_opponent_factors
from wnba.visualize import render_parlay_card, render_prediction_card, render_prop_card


@st.cache_resource
def load_model():
    return margin_model.load()


@st.cache_resource
def load_props_data():
    """Returns (player_box, opponent_factors) or (None, None) if the
    player backfill hasn't been run yet.
    """
    try:
        return load_player_boxscores(), load_opponent_factors()
    except FileNotFoundError:
        return None, None


@st.cache_data(ttl=300, show_spinner="Fetching today's games...")
def _cached_todays_games(date_str):
    return todays_games(date_str)


@st.cache_data(ttl=1800, show_spinner="Fetching live injury report...")
def _cached_injuries():
    return fetch_injuries()


@st.cache_data(ttl=600, show_spinner="Fetching game-line odds (cheap, whole-slate bulk call)...")
def _cached_game_matches(_model, bookmakers):
    """`_model` underscore-prefixed so Streamlit doesn't try to hash the
    unhashable model object. This is the CHEAP call (game lines only, one
    bulk request) -- player props are fetched separately, only for games
    the user picks. See the module docstring's cost note.
    """
    return gather_match_options(bookmakers=bookmakers)


def main() -> None:
    st.set_page_config(page_title="WNBA Predictor", layout="centered")
    st.title("WNBA Match & Player Prop Predictor")
    st.caption("Margin/total model, self-calculated Elo, bootstrap-simulated player props.")

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("No fitted model found. Run `wnba-predictor refresh` first.")
        return

    player_box, opponent_factors = load_props_data()

    tab_predict, tab_today, tab_props, tab_injuries, tab_bet = st.tabs(
        ["Predict", "Today's Games", "Player Props", "Injuries", "Bet Builder"]
    )
    with tab_predict:
        render_predict_tab(model)
    with tab_today:
        render_today_tab(model)
    with tab_props:
        render_props_tab(player_box, opponent_factors)
    with tab_injuries:
        render_injuries_tab()
    with tab_bet:
        render_bet_builder_tab(model, player_box, opponent_factors)


def render_predict_tab(model) -> None:
    teams = sorted(model.attack.keys())
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home team", teams, index=0, key="predict_home")
    with col2:
        away = st.selectbox("Away team", teams, index=min(1, len(teams) - 1), key="predict_away")
    neutral = st.checkbox("Neutral venue", value=False)

    if home == away:
        st.warning("Pick two different teams.")
        return

    win_prob = model.win_probability(home, away, neutral=neutral)
    pred_home, pred_away = model.predicted_scores(home, away, neutral=neutral)
    total_mu, _ = model.total_distribution(home, away, neutral=neutral)

    st.subheader(f"{home} vs {away}")
    c1, c2 = st.columns(2)
    c1.metric(f"{home} win", f"{win_prob*100:.1f}%")
    c2.metric(f"{away} win", f"{(1-win_prob)*100:.1f}%")
    st.write(f"**Predicted score:** {home} {pred_home:.1f} - {pred_away:.1f} {away}")
    st.write(f"**Projected total:** {total_mu:.1f}")

    with st.expander("Spread / total probability"):
        spread = st.number_input(f"{home} spread (e.g. -5.5)", value=-3.5, step=0.5)
        cover = model.prob_home_covers_spread(home, away, point=spread, neutral=neutral)
        st.write(f"P({home} covers {spread:+.1f}): {cover*100:.1f}%")
        total_line = st.number_input("Total line", value=round(total_mu * 2) / 2, step=0.5)
        over = model.prob_over(home, away, line=total_line, neutral=neutral)
        st.write(f"P(total over {total_line}): {over*100:.1f}%")

    fig = render_prediction_card(model, home, away, neutral=neutral)
    st.pyplot(fig)


def render_today_tab(model) -> None:
    date_str = st.text_input("Date (YYYY-MM-DD, blank = today)", value="")
    games = _cached_todays_games(date_str or None)
    if not games:
        st.info("No WNBA games scheduled.")
        return

    for g in games:
        line = f"**{g['away_team']} @ {g['home_team']}** [{g['status']}]"
        try:
            win_prob = model.win_probability(g["home_team"], g["away_team"], neutral=False)
            line += f" -- model: {g['home_team']} {win_prob*100:.1f}%"
        except KeyError:
            line += " -- no model rating for one of these teams yet"
        if g["completed"]:
            line += f" (final {g['home_score']}-{g['away_score']})"
        st.write(line)


def render_props_tab(player_box, opponent_factors) -> None:
    if player_box is None:
        st.warning("No player box-score data yet. Run `wnba-predictor refresh --players` first (slow: ~1 request/game).")
        return

    teams = sorted(player_box["team"].unique())
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams, index=0, key="props_team_a")
    with col2:
        team_b = st.selectbox("Team B", teams, index=min(1, len(teams) - 1), key="props_team_b")

    if team_a == team_b:
        st.warning("Pick two different teams.")
        return

    roster_a = team_roster(team_a, player_box, n_recent_games=8)
    roster_a["team_label"] = team_a
    roster_b = team_roster(team_b, player_box, n_recent_games=8)
    roster_b["team_label"] = team_b
    combined_roster = pd.concat([roster_a, roster_b], ignore_index=True)
    if combined_roster.empty:
        st.info("No recent players found for either team.")
        return

    player_choices = [f"{row.player_name} ({row.team_label})" for row in combined_roster.itertuples(index=False)]
    picked = st.selectbox("Player -- from either team", player_choices, key="props_player")
    chosen = combined_roster.iloc[player_choices.index(picked)]
    player_name, player_id, player_team = chosen["player_name"], chosen["player_id"], chosen["team_label"]
    opponent = team_b if player_team == team_a else team_a

    n_games = int((player_box["player_id"] == player_id).sum())

    try:
        sim = simulate_player_games(player_box, player_id, opponent, opponent_factors)
    except ValueError:
        st.warning(f"Not enough game history for {player_name} yet.")
        return

    st.caption(f"Projections from {n_games} games (weighted toward recent form), 10,000 simulations.")

    st.subheader(f"{player_name} -- probability board")
    st.caption(
        "Every stat at once, not just one bet slip. Lines default to just under the model's "
        "own projected mean (a stand-in, not a real sportsbook line) -- edit any line to check "
        "a real number and the probabilities below update immediately."
    )
    board_stats = STAT_COLUMNS + ["pra", "pr", "pa", "ra", "sb"]
    line_df = pd.DataFrame({
        "Stat": [STAT_LABELS[s] for s in board_stats],
        "Projected": [round(float(sim[s].mean()), 1) for s in board_stats],
        "Line": [round(float(sim[s].mean())) - 0.5 for s in board_stats],
    })
    edited_lines = st.data_editor(
        line_df,
        column_config={"Line": st.column_config.NumberColumn(step=0.5, help="Edit to check a real sportsbook line")},
        disabled=["Stat", "Projected"],
        hide_index=True, use_container_width=True, key="props_board_lines",
    )

    board_results = []
    for stat, row in zip(board_stats, edited_lines.itertuples(index=False)):
        p_over = prob_over(sim, stat, row.Line)
        # ProgressColumn's `format` is applied to the raw cell value, not auto-scaled --
        # store 0-100 (not 0-1) so "%.0f%%" actually prints e.g. "78%" instead of "1%".
        board_results.append({"Stat": row.Stat, "Line": row.Line, "P(Over)": p_over * 100, "P(Under)": (1 - p_over) * 100})
    st.dataframe(
        pd.DataFrame(board_results),
        column_config={
            "P(Over)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
            "P(Under)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        },
        hide_index=True, use_container_width=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("P(Double-Double)", f"{prob_double_double(sim)*100:.0f}%")
    c2.metric("P(Triple-Double)", f"{prob_triple_double(sim)*100:.0f}%")

    st.subheader("Prop card")
    card_stat = st.selectbox("Stat to feature on the card", board_stats, format_func=lambda s: STAT_LABELS[s], key="props_card_stat")
    card_line = round(float(sim[card_stat].mean())) - 0.5
    fig = render_prop_card(player_name, opponent, sim, card_stat, line=card_line)
    st.pyplot(fig)


def render_injuries_tab() -> None:
    injuries = _cached_injuries()
    if injuries.empty:
        st.info("No injuries currently reported.")
        return
    teams = ["All teams"] + sorted(injuries["team"].unique())
    team_choice = st.selectbox("Team", teams, key="injuries_team")
    shown = injuries if team_choice == "All teams" else injuries[injuries["team"] == team_choice]
    for row in shown.itertuples(index=False):
        st.write(f"**[{row.team}] {row.player_name}** ({row.position}) -- {row.status}")
        st.caption(row.comment)


def render_bet_builder_tab(model, player_box, opponent_factors) -> None:
    st.caption(
        "Pick which games interest you, optionally give a target payout, and it searches only "
        "within those games for the best fit. **Not a guarantee - for analysis/entertainment only.**"
    )
    st.caption(
        "⚠️ Player props are fetched live and cost real Odds-API credits per game selected "
        "(the free tier is limited -- see this file's docstring). Game lines below are cheap."
    )

    book_choice = st.selectbox("Sportsbook", ["Average across all"] + KNOWN_BOOKMAKERS)
    bookmakers = None if book_choice == "Average across all" else book_choice

    if st.button("Load today's games and odds (one cheap bulk call)"):
        st.session_state["wnba_odds_loaded"] = True
    if not st.session_state.get("wnba_odds_loaded"):
        st.info("Click above to fetch the current game list and odds -- nothing is fetched automatically on page load.")
        return

    matches = _cached_game_matches(model, bookmakers)
    matches = [m for m in matches if m.legs]
    if not matches:
        st.warning("No upcoming games found.")
        return

    match_labels = [f"{m.home_team} vs {m.away_team} ({m.date})" for m in matches]
    match_by_label = dict(zip(match_labels, matches))
    chosen_labels = st.multiselect("Which games do you want in your parlay?", match_labels)

    include_props = st.checkbox("Include player props for chosen games (uses odds credits)", value=False)

    if chosen_labels:
        st.caption("Best game-line pick per chosen game, for reference:")
        for label in chosen_labels:
            b = match_by_label[label].best_leg
            st.write(f"- **{label}**: {b.pick_label} {b.odds:.2f}x -- {b.reasoning}")

    use_target = st.checkbox("Target a specific payout", value=False)
    target = st.number_input("Target payout (e.g. 5 = 5x)", min_value=1.1, value=5.0, step=0.5) if use_target else None
    stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=5.0)

    if st.button("Build my parlay", type="primary"):
        if not chosen_labels:
            st.warning("Pick at least one game first.")
            return

        chosen_matches = [match_by_label[label] for label in chosen_labels]
        if include_props and player_box is not None:
            with st.spinner("Fetching player props for chosen games..."):
                chosen_matches = [add_player_prop_legs(m, player_box, opponent_factors, bookmakers=bookmakers) for m in chosen_matches]

        if target is not None:
            selected = find_best_combo(chosen_matches, target)
            actual = 1.0
            for leg in selected:
                actual *= leg.odds
            if abs(actual - target) / target > 0.15:
                st.info(f"Closest available is {actual:.2f}x -- can't get near {target:.1f}x without adding more games or accepting a longshot leg.")
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
