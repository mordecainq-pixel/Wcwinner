"""Command-line interface: predict a matchup, check today's slate, project
player props, or refresh data/refit models.

Requires `wnba-predictor refresh` to have been run at least once so a fitted
model exists in data/wnba_processed/. Player props additionally require
`wnba-predictor refresh --players` (much slower: one HTTP request per game).
"""
from __future__ import annotations

import argparse

import pandas as pd

from wnba.betbuilder import add_player_prop_legs, combine_legs, find_best_combo, gather_match_options, out_player_ids_for_team
from wnba.data import espn_ingest
from wnba.data.espn_injuries import fetch_injuries, team_injuries
from wnba.data.espn_live import team_roster, todays_games
from wnba.data.espn_player_ingest import load_player_availability, load_player_boxscores
from wnba.data.odds_client import game_market_probabilities
from wnba.model import margin_model
from wnba.model.player_model import (
    STAT_COLUMNS,
    STAT_LABELS,
    prob_double_double,
    prob_over,
    prob_triple_double,
    simulate_player_games,
)
from wnba.pipeline import load_opponent_factors, run_full_refresh
from wnba.validate.backtest import run_backtest


def cmd_predict(args: argparse.Namespace) -> None:
    model = margin_model.load()
    neutral = not args.home_advantage
    pred_home, pred_away = model.predicted_scores(args.home, args.away, neutral=neutral)
    win_prob = model.win_probability(args.home, args.away, neutral=neutral)
    total_mu, total_sd = model.total_distribution(args.home, args.away, neutral=neutral)

    venue = " (neutral venue)" if neutral else f" ({args.home} at home)"
    print(f"\n{args.home} vs {args.away}{venue}")
    print(f"Predicted score:  {args.home} {pred_home:.1f} - {pred_away:.1f} {args.away}")
    print(f"Win probability:  {args.home} {win_prob*100:.1f}% / {args.away} {(1-win_prob)*100:.1f}%")
    print(f"Projected total:  {total_mu:.1f} (std {total_sd:.1f})")

    if args.spread is not None:
        cover = model.prob_home_covers_spread(args.home, args.away, point=args.spread, neutral=neutral)
        print(f"P({args.home} covers {args.spread:+.1f}): {cover*100:.1f}%")
    if args.total_line is not None:
        over = model.prob_over(args.home, args.away, line=args.total_line, neutral=neutral)
        print(f"P(total over {args.total_line}): {over*100:.1f}%")


def cmd_today(args: argparse.Namespace) -> None:
    model = margin_model.load()
    games = todays_games(args.date)
    if not games:
        print(f"No WNBA games scheduled for {args.date or 'today'}.")
        return

    print(f"\nWNBA slate for {args.date or 'today'}:\n")
    for g in games:
        line = f"  {g['away_team']} @ {g['home_team']}  [{g['status']}]"
        try:
            win_prob = model.win_probability(g["home_team"], g["away_team"], neutral=False)
            line += f"  -- model: {g['home_team']} {win_prob*100:.1f}%"
        except KeyError:
            line += "  -- model: no rating for one of these teams yet"
        if g["completed"]:
            line += f"  (final {g['home_score']}-{g['away_score']})"
        print(line)


def cmd_roster(args: argparse.Namespace) -> None:
    player_box = load_player_boxscores()
    roster = team_roster(args.team, player_box, n_recent_games=args.recent_games)
    if roster.empty:
        print(f"No recent players found for {args.team!r}. Check the team name / run `refresh --players`.")
        return
    print(f"\n{args.team} -- players active in the last {args.recent_games} games:\n")
    print(roster.to_string(index=False))


def _resolve_player_id(player_box: pd.DataFrame, name: str) -> tuple[str, str]:
    matches = player_box[player_box["player_name"].str.lower() == name.lower()]
    if matches.empty:
        matches = player_box[player_box["player_name"].str.lower().str.contains(name.lower())]
    if matches.empty:
        raise SystemExit(f"No player found matching {name!r}.")
    player_id = matches.iloc[-1]["player_id"]
    resolved_name = matches.iloc[-1]["player_name"]
    return player_id, resolved_name


def cmd_props(args: argparse.Namespace) -> None:
    player_box = load_player_boxscores()
    factors = load_opponent_factors()
    player_id, resolved_name = _resolve_player_id(player_box, args.player)

    try:
        availability = load_player_availability()
        out_ids = out_player_ids_for_team(fetch_injuries(), args.opponent)
    except FileNotFoundError:
        availability, out_ids = None, set()

    sim = simulate_player_games(
        player_box, player_id, args.opponent, factors, n_sims=args.sims,
        out_player_ids=out_ids, availability=availability,
    )
    n_games = (player_box["player_id"] == player_id).sum()

    print(f"\n{resolved_name} vs {args.opponent} -- projections from last {n_games} games ({args.sims} simulations):\n")
    if out_ids:
        print(f"  (adjusted for {len(out_ids)} player(s) listed Out for {args.opponent}, where enough history exists)")
    for stat in STAT_COLUMNS:
        label = STAT_LABELS[stat]
        print(f"  {label:<16} mean {sim[stat].mean():5.1f}  median {sim[stat].median():5.1f}  std {sim[stat].std():4.1f}")
    for stat in ("pra", "pr", "pa", "ra", "sb"):
        label = STAT_LABELS[stat]
        print(f"  {label:<16} mean {sim[stat].mean():5.1f}  median {sim[stat].median():5.1f}  std {sim[stat].std():4.1f}")
    print(f"\n  P(double-double): {prob_double_double(sim)*100:.1f}%")
    print(f"  P(triple-double): {prob_triple_double(sim)*100:.1f}%")

    if args.stat and args.line is not None:
        p = prob_over(sim, args.stat, args.line)
        print(f"\n  P({args.stat} over {args.line}): {p*100:.1f}%  (under: {(1-p)*100:.1f}%)")


def cmd_market(args: argparse.Namespace) -> None:
    model = margin_model.load()
    df = game_market_probabilities(bookmakers=args.bookmakers)
    if df.empty:
        print("No upcoming games with market odds found.")
        return

    print("\nModel vs. market (The Odds API), upcoming WNBA games:\n")
    rows = []
    for row in df.itertuples(index=False):
        try:
            model_home_prob = model.win_probability(row.home_team, row.away_team, neutral=False)
        except KeyError:
            continue
        rows.append({
            "matchup": f"{row.away_team} @ {row.home_team}",
            "model_home_%": model_home_prob * 100,
            "market_home_%": row.market_home_prob * 100 if row.market_home_prob else None,
            "edge_%": (model_home_prob - row.market_home_prob) * 100 if row.market_home_prob else None,
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


def cmd_injuries(args: argparse.Namespace) -> None:
    injuries = fetch_injuries()
    if args.team:
        injuries = team_injuries(args.team, injuries=injuries)
    if injuries.empty:
        print("No injuries reported" + (f" for {args.team}" if args.team else "") + ".")
        return
    print(f"\nWNBA injury report{' -- ' + args.team if args.team else ''}:\n")
    for row in injuries.itertuples(index=False):
        print(f"  [{row.team}] {row.player_name} ({row.position}) -- {row.status}: {row.comment}")


def _print_betbuilder_result(result) -> None:
    print(f"\n{len(result.legs)}-leg parlay:\n")
    for i, leg in enumerate(result.legs, start=1):
        print(f"  {i}. [{leg.market_type:9s}] {leg.pick_label:45s} {leg.odds:6.2f}x   {leg.reasoning}   ({leg.date})")
    print(f"\nCombined odds: {result.combined_odds:.2f}x")
    print(f"Model's estimated chance all legs hit: {result.combined_model_prob*100:.1f}%")
    print(f"Stake ${result.stake:.2f} -> payout ${result.payout:.2f} (profit ${result.profit:.2f})")
    print("\nNot a guarantee. For analysis/entertainment only.")


def cmd_betbuilder(args: argparse.Namespace) -> None:
    bookmakers = args.bookmakers
    if bookmakers is None and not args.no_prompt_bookmakers:
        raw = input(
            "Sportsbook to use, e.g. fanduel, draftkings, espnbet "
            "(blank = average across all available; comma-separate for a few): "
        ).strip()
        bookmakers = raw or None

    matches = gather_match_options(bookmakers=bookmakers)
    matches = [m for m in matches if m.legs]
    if not matches:
        print("No upcoming games found.")
        return

    print("\nAvailable games (game lines only shown here -- player props are fetched only for what you pick, since they're expensive on the free odds tier):\n")
    for i, m in enumerate(matches, start=1):
        b = m.best_leg
        print(f"  {i:>2}. {m.home_team} vs {m.away_team} ({m.date}) -- best: {b.pick_label} {b.odds:.2f}x, {b.reasoning}")

    if args.games:
        chosen_idx = [int(x) for x in args.games.split(",")]
    else:
        raw = input("\nWhich games do you want in your parlay? (comma-separated numbers, or 'all'): ").strip()
        chosen_idx = list(range(1, len(matches) + 1)) if raw.lower() == "all" else [int(x) for x in raw.split(",")]
    chosen_matches = [matches[i - 1] for i in chosen_idx]

    if not args.no_props:
        try:
            player_box = load_player_boxscores()
            factors = load_opponent_factors()
            try:
                availability = load_player_availability()
                injuries = fetch_injuries()
            except FileNotFoundError:
                availability, injuries = None, None
            chosen_matches = [
                add_player_prop_legs(m, player_box, factors, bookmakers=bookmakers, availability=availability, injuries=injuries)
                for m in chosen_matches
            ]
        except FileNotFoundError:
            print("\n(No player box-score data yet -- run `refresh --players` for prop legs. Continuing with game lines only.)")

    target = args.target
    if target is None and not args.no_prompt_target:
        raw = input("Target payout multiple, e.g. 5 for 5x (blank = just use the single best pick per game): ").strip()
        target = float(raw) if raw else None

    if target is not None:
        selected = find_best_combo(chosen_matches, target)
        actual = 1.0
        for leg in selected:
            actual *= leg.odds
        if abs(actual - target) / target > 0.15:
            print(f"\nHeads up: the closest combination available is {actual:.2f}x -- can't get near {target:.1f}x without adding more games or picking a longshot leg on purpose.")
    else:
        selected = [m.best_leg for m in chosen_matches]

    stake = args.stake
    if stake is None:
        raw = input("Stake amount ($, blank = 10): ").strip()
        stake = float(raw) if raw else 10.0

    result = combine_legs(selected, stake=stake)
    _print_betbuilder_result(result)


def cmd_backtest(args: argparse.Namespace) -> None:
    results = espn_ingest.load_results()
    bt = run_backtest(results, cutoff_date=args.cutoff, max_test_games=args.max_test_games)

    print(f"Train: {bt['n_train']} games through {bt['cutoff_date'].date()}")
    print(f"Test:  {bt['n_test']} games after cutoff\n")
    print(f"{'Metric':<12}{'Model':>10}{'Elo-only baseline':>20}")
    print(f"{'Log-loss':<12}{bt['model_log_loss']:>10.4f}{bt['baseline_log_loss']:>20.4f}")
    print(f"{'Brier':<12}{bt['model_brier']:>10.4f}{bt['baseline_brier']:>20.4f}")


def cmd_refresh(args: argparse.Namespace) -> None:
    run_full_refresh(refresh_players=args.players)


def main() -> None:
    parser = argparse.ArgumentParser(prog="wnba-predictor", description="WNBA margin/total model + player props")
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Predict a single matchup")
    p_predict.add_argument("home")
    p_predict.add_argument("away")
    p_predict.add_argument("--home-advantage", action="store_true", help="Non-neutral game with `home` at home")
    p_predict.add_argument("--spread", type=float, default=None, help="e.g. -5.5 means home must win by 6+")
    p_predict.add_argument("--total-line", type=float, default=None)
    p_predict.set_defaults(func=cmd_predict)

    p_today = sub.add_parser("today", help="List today's (or a given date's) games with model win probabilities")
    p_today.add_argument("--date", default=None, help="YYYY-MM-DD, default: today")
    p_today.set_defaults(func=cmd_today)

    p_roster = sub.add_parser("roster", help="Show a team's recently active players")
    p_roster.add_argument("team")
    p_roster.add_argument("--recent-games", type=int, default=5)
    p_roster.set_defaults(func=cmd_roster)

    p_props = sub.add_parser("props", help="Project a player's stat lines against an opponent")
    p_props.add_argument("player")
    p_props.add_argument("--opponent", required=True)
    p_props.add_argument("--stat", default=None, help="e.g. points, rebounds, pra, sb")
    p_props.add_argument("--line", type=float, default=None)
    p_props.add_argument("--sims", type=int, default=10000)
    p_props.set_defaults(func=cmd_props)

    p_market = sub.add_parser("market", help="Compare model probabilities to live betting odds")
    p_market.add_argument("--bookmakers", default=None, help="e.g. fanduel or fanduel,draftkings (default: average across all)")
    p_market.set_defaults(func=cmd_market)

    p_injuries = sub.add_parser("injuries", help="Show the live league-wide (or one team's) injury report")
    p_injuries.add_argument("--team", default=None)
    p_injuries.set_defaults(func=cmd_injuries)

    p_bet = sub.add_parser("betbuilder", help="Pick games and (optionally) a target payout; game + player-prop legs")
    p_bet.add_argument("--bookmakers", default=None, help="e.g. fanduel or fanduel,draftkings (default: average across all)")
    p_bet.add_argument("--no-prompt-bookmakers", action="store_true")
    p_bet.add_argument("--games", default=None, help="Comma-separated game numbers to include (skips the interactive prompt)")
    p_bet.add_argument("--target", type=float, default=None)
    p_bet.add_argument("--no-prompt-target", action="store_true")
    p_bet.add_argument("--no-props", action="store_true", help="Game legs only, skip player props")
    p_bet.add_argument("--stake", type=float, default=None)
    p_bet.set_defaults(func=cmd_betbuilder)

    p_backtest = sub.add_parser("backtest", help="Backtest against held-out real games")
    p_backtest.add_argument("--cutoff", default="2024-07-01")
    p_backtest.add_argument("--max-test-games", type=int, default=None)
    p_backtest.set_defaults(func=cmd_backtest)

    p_refresh = sub.add_parser("refresh", help="Re-download data and refit models")
    p_refresh.add_argument("--players", action="store_true", help="Also refresh player box scores (slow: ~1 request/game)")
    p_refresh.set_defaults(func=cmd_refresh)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
