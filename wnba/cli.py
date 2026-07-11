"""Command-line interface: predict a matchup, check today's slate, project
player props, or refresh data/refit models.

Requires `wnba-predictor refresh` to have been run at least once so a fitted
model exists in data/wnba_processed/. Player props additionally require
`wnba-predictor refresh --players` (much slower: one HTTP request per game).
"""
from __future__ import annotations

import argparse

import pandas as pd

from wnba.data import espn_ingest
from wnba.data.espn_live import team_roster, todays_games
from wnba.data.espn_player_ingest import load_player_boxscores
from wnba.model import margin_model
from wnba.model.player_model import (
    STAT_COLUMNS,
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

    sim = simulate_player_games(player_box, player_id, args.opponent, factors, n_sims=args.sims)
    n_games = (player_box["player_id"] == player_id).sum()

    print(f"\n{resolved_name} vs {args.opponent} -- projections from last {n_games} games ({args.sims} simulations):\n")
    combo_labels = {"pra": "PTS+REB+AST", "pr": "PTS+REB", "pa": "PTS+AST", "ra": "REB+AST", "sb": "STL+BLK"}
    for stat in STAT_COLUMNS:
        print(f"  {stat:<10} mean {sim[stat].mean():5.1f}  median {sim[stat].median():5.1f}  std {sim[stat].std():4.1f}")
    for stat, label in combo_labels.items():
        print(f"  {label:<10} mean {sim[stat].mean():5.1f}  median {sim[stat].median():5.1f}  std {sim[stat].std():4.1f}")
    print(f"\n  P(double-double): {prob_double_double(sim)*100:.1f}%")
    print(f"  P(triple-double): {prob_triple_double(sim)*100:.1f}%")

    if args.stat and args.line is not None:
        p = prob_over(sim, args.stat, args.line)
        print(f"\n  P({args.stat} over {args.line}): {p*100:.1f}%  (under: {(1-p)*100:.1f}%)")


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
