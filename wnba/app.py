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

from wnba import admin, status as refresh_status
from wnba.betbuilder import KNOWN_BOOKMAKERS, add_player_prop_legs, combine_legs, find_best_combo, gather_match_options
from wnba.data.espn_injuries import fetch_injuries
from wnba.data.espn_live import team_roster, todays_games
from wnba.data.espn_player_ingest import load_player_availability, load_player_boxscores
from wnba.data.odds_client import game_market_probabilities
from wnba.model import margin_model
from wnba.model.player_model import STAT_COLUMNS, STAT_LABELS, prob_double_double, prob_over, prob_triple_double, simulate_player_games
from wnba.pipeline import load_opponent_factors
from wnba.value_scan import load_cached_scan, save_scan, scan_bookmaker_value
from wnba.visualize import render_parlay_card, render_prediction_card, render_prop_card


@st.cache_resource
def load_model():
    return margin_model.load()


@st.cache_resource
def load_props_data():
    """Returns (player_box, opponent_factors, availability), or (None, None,
    None) if the player backfill hasn't been run yet. `availability` (used
    for the injury-aware opponent adjustment) comes back None on its own if
    player data was last refreshed before that feature existed -- props and
    Bet Builder still work, just without the adjustment, rather than
    breaking outright.
    """
    try:
        player_box, opponent_factors = load_player_boxscores(), load_opponent_factors()
    except FileNotFoundError:
        return None, None, None
    try:
        availability = load_player_availability()
    except FileNotFoundError:
        availability = None
    return player_box, opponent_factors, availability


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


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_simulate(_player_box, _opponent_factors, player_id, opponent, out_ids=frozenset(), _availability=None):
    """Without this cache, simulate_player_games() (an unseeded random
    bootstrap) reran on every single widget interaction anywhere on the
    page -- Streamlit reruns the whole script on any rerun, including the
    one triggered by editing a cell in the probability board below. That
    meant the board's underlying data changed out from under the editor on
    every keystroke, so an edited Line value just got silently overwritten
    by a freshly-randomized default instead of sticking. Caching per
    (player_id, opponent, out_ids) keeps the simulation -- and the editor's
    base data -- stable across reruns, which is what makes edits actually
    hold. `_player_box`/`_opponent_factors`/`_availability`
    underscore-prefixed so Streamlit doesn't try to hash the (large)
    DataFrames themselves; `out_ids` is a frozenset (hashable) so it can
    still be a real cache key -- a change in who's out should invalidate it.
    """
    return simulate_player_games(
        _player_box, player_id, opponent, _opponent_factors,
        out_player_ids=out_ids, availability=_availability,
    )


def render_main_app() -> None:
    st.title("WNBA Match & Player Prop Predictor")
    st.caption("Margin/total model, self-calculated Elo, bootstrap-simulated player props.")

    is_admin = admin.is_admin()
    current_status = refresh_status.read_status()
    if current_status["state"] == "updating" and not is_admin:
        render_updating_banner(current_status)
        return

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("No fitted model found. Run `wnba-predictor refresh` first.")
        return

    player_box, opponent_factors, availability = load_props_data()

    if current_status["state"] == "updating":
        st.info("Data refresh in progress (visible to you as admin; hidden from other visitors until it finishes). See below for details.")

    tabs = ["Predict", "Today's Games", "Player Props", "Injuries"]
    if is_admin:
        tabs += ["Bet Builder", "Value Scan", "Admin"]
    rendered = st.tabs(tabs)

    with rendered[0]:
        render_predict_tab(model)
    with rendered[1]:
        render_today_tab(model)
    with rendered[2]:
        render_props_tab(player_box, opponent_factors, availability)
    with rendered[3]:
        render_injuries_tab()
    if is_admin:
        with rendered[4]:
            render_bet_builder_tab(model, player_box, opponent_factors, availability)
        with rendered[5]:
            render_value_scan_tab(model, player_box, opponent_factors, availability)
        with rendered[6]:
            render_admin_tab(current_status)

    finished = current_status.get("finished_at")
    if finished:
        st.caption(f"Data last updated: {finished[:16].replace('T', ' ')} UTC")


def main() -> None:
    st.set_page_config(page_title="WNBA Predictor", layout="centered", page_icon="🏀")

    main_page = st.Page(render_main_app, title="WNBA Predictor", url_path="", default=True)
    admin_page = st.Page(
        lambda: admin.render_admin_login_page(main_page),
        title="Admin",
        url_path="adminlogs",
        visibility="hidden",
    )
    # position="hidden" means no nav widget is shown to anyone at all --
    # admin_page is reachable only by knowing the /adminlogs URL directly.
    pg = st.navigation([main_page, admin_page], position="hidden")
    pg.run()


def render_updating_banner(current_status: dict) -> None:
    eta = current_status.get("eta_minutes") or 25
    elapsed = refresh_status.minutes_since_started(current_status)
    remaining = max(1, round(eta - elapsed)) if elapsed is not None else eta

    st.info(
        f"""
**Data Refresh In Progress**

This site's statistics and projections are currently being updated with the latest games and results.
This process typically takes {eta} minutes; an estimated **{remaining} minutes** remain.

Please check back shortly -- thank you for your patience.
"""
    )


def render_admin_tab(current_status: dict) -> None:
    st.subheader("Data status")
    state = current_status["state"]
    if state == "updating":
        elapsed = refresh_status.minutes_since_started(current_status)
        st.warning(f"A refresh is currently running (started {elapsed:.0f} min ago, ETA {current_status.get('eta_minutes')} min).")
    else:
        finished = current_status.get("finished_at")
        st.success(f"Idle. Last refresh finished: {finished or 'unknown'}.")

    try:
        results_freshness = espn_ingest_freshness()
        st.write(f"**Team results:** {results_freshness['rows_total']} games, most recent {results_freshness['max_date'].date()}")
    except FileNotFoundError:
        st.write("**Team results:** not loaded yet.")

    player_box, _, _ = load_props_data()
    if player_box is not None:
        st.write(f"**Player box scores:** {len(player_box)} rows, most recent {player_box['date'].max().date()}")
    else:
        st.write("**Player box scores:** not loaded yet.")

    st.divider()
    st.subheader("Trigger a data refresh")
    st.caption(
        "Runs the same GitHub Actions job as the daily automatic refresh (not run inside this app "
        "directly -- it's slow and rate-limited against ESPN). Takes ~15-25 minutes; this site "
        "redeploys automatically once it's done."
    )
    if st.button("Refresh data now", type="primary"):
        ok, message = admin.trigger_refresh_workflow()
        (st.success if ok else st.error)(message)


def espn_ingest_freshness():
    from wnba.data import espn_ingest
    return espn_ingest.data_freshness()


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


def render_props_tab(player_box, opponent_factors, availability=None) -> None:
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

    # Injury-aware opponent adjustment exists (model/player_model.py) but is
    # deliberately NOT wired in live here -- backtested and found only a
    # weak, mostly not-statistically-significant effect (see
    # validate/player_backtest.py's compare_injury_adjustment), so it stays
    # off by default rather than shipped as if it were a proven improvement.
    try:
        sim = _cached_simulate(player_box, opponent_factors, player_id, opponent)
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


def render_bet_builder_tab(model, player_box, opponent_factors, availability=None) -> None:
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
        # Injury-aware opponent adjustment intentionally not wired in here --
        # see render_props_tab's comment for why.
        if include_props and player_box is not None:
            with st.spinner("Fetching player props for chosen games..."):
                chosen_matches = [
                    add_player_prop_legs(m, player_box, opponent_factors, bookmakers=bookmakers)
                    for m in chosen_matches
                ]

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


def render_value_scan_tab(model, player_box, opponent_factors, availability=None) -> None:
    st.caption(
        "Scan ONE sportsbook's live player-prop lines against the model's own projections and see "
        "which ones show the biggest gap. **Not a guarantee - for analysis/entertainment only.**"
    )
    st.caption(
        "⚠️ Scanning costs real Odds-API credits per game selected (same market as Bet Builder's "
        "player props). Results are cached on disk -- reopening this tab does NOT re-spend credits, "
        "only the 'Scan now' button below does."
    )

    if player_box is None:
        st.warning("No player box-score data yet. Run `wnba-predictor refresh --players` first.")
        return

    book_choice = st.selectbox("Sportsbook to scan", KNOWN_BOOKMAKERS, key="value_scan_book")

    if st.button("Load today's games and odds (one cheap bulk call)", key="value_scan_load_games"):
        st.session_state["value_scan_odds_loaded"] = True
    if not st.session_state.get("value_scan_odds_loaded"):
        st.info("Click above to fetch the current game list -- nothing is fetched automatically on page load.")
    else:
        matches = _cached_game_matches(model, book_choice)
        matches = [m for m in matches if m.legs]
        if not matches:
            st.warning("No upcoming games found.")
        else:
            match_labels = [f"{m.home_team} vs {m.away_team} ({m.date})" for m in matches]
            match_by_label = dict(zip(match_labels, matches))
            chosen_labels = st.multiselect("Which games do you want to scan for value?", match_labels, key="value_scan_games")

            if st.button(f"Scan {book_choice} value now (uses Odds API credits)", type="primary", key="value_scan_run"):
                if not chosen_labels:
                    st.warning("Pick at least one game first.")
                else:
                    chosen_matches = [match_by_label[label] for label in chosen_labels]
                    # Injury-aware opponent adjustment intentionally not
                    # wired in here -- see render_props_tab's comment for why.
                    with st.spinner(f"Scanning {book_choice} player props for {len(chosen_matches)} game(s)..."):
                        rows = scan_bookmaker_value(chosen_matches, player_box, opponent_factors, book_choice)
                        save_scan(rows, book_choice)
                    st.success(f"Scanned {len(chosen_matches)} game(s) -- found {len(rows)} prop/milestone line(s).")

    st.divider()
    cached = load_cached_scan()
    if not cached:
        st.info("No cached scan yet -- pick a book and games above, then scan.")
        return

    fetched_dt = pd.to_datetime(cached["fetched_at"])
    age_min = (pd.Timestamp.now(tz="UTC") - fetched_dt).total_seconds() / 60
    st.caption(
        f"Showing cached scan: **{cached['bookmaker']}**, fetched {age_min:.0f} min ago "
        f"({fetched_dt.strftime('%Y-%m-%d %H:%M UTC')}). Odds move -- re-scan if this looks stale."
    )

    rows = cached["rows"]
    if not rows:
        st.info("That scan found no matching prop/milestone lines.")
        return

    df = pd.DataFrame(rows)
    df["pick"] = df.apply(
        lambda r: f"{r['best_side'].capitalize()} {r['book_line']}" if r["book_line"] is not None else "Yes", axis=1,
    )
    df["ev_pct"] = df["best_ev"] * 100
    shown = df[["player_name", "stat", "date", "home_team", "away_team", "book_line", "model_mean", "gap", "pick", "ev_pct"]].rename(columns={
        "player_name": "Player", "stat": "Stat", "date": "Date", "home_team": "Home", "away_team": "Away",
        "book_line": "Book Line", "model_mean": "Model Mean", "gap": "Gap (Model - Book)", "pick": "Best Side", "ev_pct": "EV",
    })
    st.caption("Ranked by EV (accounts for the book's actual payout odds), not the raw gap -- a big gap with bad odds isn't necessarily good value.")
    st.dataframe(
        shown.sort_values("EV", ascending=False),
        column_config={
            "EV": st.column_config.NumberColumn(format="%.0f%%"),
            "Gap (Model - Book)": st.column_config.NumberColumn(format="%+.1f"),
            "Model Mean": st.column_config.NumberColumn(format="%.1f"),
        },
        hide_index=True, use_container_width=True,
    )


if __name__ == "__main__":
    main()
