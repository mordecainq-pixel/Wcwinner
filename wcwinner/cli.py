"""Command-line interface: predict a matchup, run the bracket Monte Carlo,
backtest, or compare against the market.

Requires `python scripts/refresh_data.py` to have been run at least once so a
fitted model exists in data/processed/.
"""
from __future__ import annotations

import argparse
import pickle

from wcwinner.betbuilder import build_parlay, gather_candidate_legs
from wcwinner.config import ELO_RATINGS_PATH
from wcwinner.data.football_data_client import bracket_state
from wcwinner.model import dixon_coles
from wcwinner.simulate.bracket import simulate_tournament
from wcwinner.simulate.match import predict_match
from wcwinner.validate.backtest import run_backtest
from wcwinner.validate.calibration import calibration_report
from wcwinner.validate.market_compare import compare_to_market
from wcwinner.visualize import render_parlay_card, render_prediction_card


def _load_model_and_elo():
    model = dixon_coles.load()
    with open(ELO_RATINGS_PATH, "rb") as f:
        elo_ratings = pickle.load(f)
    return model, elo_ratings


def _print_scoreline_heatmap(matrix, home_team, away_team, max_goals=6) -> None:
    print(f"\nScoreline probabilities (rows = {home_team} goals, cols = {away_team} goals):")
    print("      " + "".join(f"{j:>7d}" for j in range(max_goals + 1)))
    for i in range(max_goals + 1):
        row = "".join(f"{matrix[i, j] * 100:6.2f}%" for j in range(max_goals + 1))
        print(f"  {i:>2d} {row}")


def cmd_predict(args: argparse.Namespace) -> None:
    model, elo_ratings = _load_model_and_elo()
    neutral = not args.home_advantage
    pred = predict_match(
        model, args.home, args.away, neutral=neutral, knockout=args.knockout, elo_ratings=elo_ratings,
        home_strength_multiplier=args.home_strength, away_strength_multiplier=args.away_strength,
    )

    venue = " (neutral venue)" if neutral else f" ({args.home} at home)"
    print(f"\n{args.home} vs {args.away}{venue}")
    print(f"Expected goals:      {args.home} {pred['expected_goals_home']:.2f} - {pred['expected_goals_away']:.2f} {args.away}")
    print(f"Win / Draw / Loss:   {pred['p_home_win']*100:.1f}% / {pred['p_draw']*100:.1f}% / {pred['p_away_win']*100:.1f}%")
    print(f"Both teams to score: {pred['p_btts']*100:.1f}%")
    i, j = pred["most_likely_score"]
    print(f"Most likely score:   {i}-{j} ({pred['most_likely_score_prob']*100:.1f}%)")

    if args.knockout:
        ko = pred["knockout"]
        print(f"\nKnockout advancement: {args.home} {ko['p_home_advances']*100:.1f}% vs {args.away} {ko['p_away_advances']*100:.1f}%")
        print(
            f"  regulation W/D/L: {ko['p_home_regulation_win']*100:.1f}% / "
            f"{ko['p_draw_regulation']*100:.1f}% / {ko['p_away_regulation_win']*100:.1f}%"
            f" -- if drawn, ET/penalty tiebreak favors {args.home} {ko['p_home_tiebreak_win']*100:.1f}%"
        )

    _print_scoreline_heatmap(pred["score_matrix"], args.home, args.away)


def cmd_bracket(args: argparse.Namespace) -> None:
    model, elo_ratings = _load_model_and_elo()
    bs = bracket_state()
    result = simulate_tournament(model, elo_ratings, bracket=bs, n_iterations=args.iterations, seed=args.seed)
    percent = result.head(args.top) * 100
    print(f"\nMonte Carlo bracket simulation ({args.iterations} iterations, current matchday {bs['current_matchday']}):\n")
    print(percent.to_string(float_format=lambda x: f"{x:.1f}%"))


def cmd_backtest(args: argparse.Namespace) -> None:
    from wcwinner.data.kaggle_ingest import load_results

    df = load_results()
    bt = run_backtest(df, cutoff_date=args.cutoff, max_test_matches=args.max_test_matches)

    print(f"Train: {bt['n_train_matches']} matches through {bt['cutoff_date'].date()}")
    print(f"Test:  {bt['n_test_matches']} matches after cutoff\n")
    print(f"{'Metric':<12}{'Model':>10}{'Elo-only baseline':>20}")
    print(f"{'Log-loss':<12}{bt['model_log_loss']:>10.4f}{bt['baseline_log_loss']:>20.4f}")
    print(f"{'Brier':<12}{bt['model_brier']:>10.4f}{bt['baseline_brier']:>20.4f}")

    cal = calibration_report(bt["model_probs"], bt["true_idx"])
    print("\nHome-win calibration (predicted vs. observed frequency per bin):")
    print(cal["home_win"].to_string(index=False))


def cmd_market(args: argparse.Namespace) -> None:
    model, _ = _load_model_and_elo()
    df = compare_to_market(model)
    print("\nModel vs. market (The Odds API), upcoming WC 2026 fixtures:\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x*100:.1f}%"))


def cmd_card(args: argparse.Namespace) -> None:
    model, elo_ratings = _load_model_and_elo()
    fig = render_prediction_card(
        model, args.home, args.away, elo_ratings,
        neutral=not args.home_advantage, knockout=args.knockout,
        stage_label=args.stage, save_path=args.out,
    )
    print(f"Saved card to {args.out}")


def cmd_betbuilder(args: argparse.Namespace) -> None:
    payout = args.payout
    if payout is None:
        payout = float(input("Target payout multiple (e.g. 5 for 5x): ").strip())

    n_legs = args.legs
    if n_legs is None:
        raw = input("How many legs? (blank = let it choose): ").strip()
        n_legs = int(raw) if raw else None

    stake = args.stake
    if stake is None:
        raw = input("Stake amount ($, blank = 10): ").strip()
        stake = float(raw) if raw else 10.0

    date_from, date_to = args.date_from, args.date_to
    if date_from is None and date_to is None and not args.no_prompt_dates:
        raw = input("Date range as YYYY-MM-DD,YYYY-MM-DD (blank = all upcoming): ").strip()
        if raw:
            date_from, date_to = [p.strip() for p in raw.split(",", 1)]

    model, elo_ratings = _load_model_and_elo()
    legs = gather_candidate_legs(model, elo_ratings, date_from, date_to)
    if not legs:
        print("No upcoming matches found in that range.")
        return

    result = build_parlay(legs, target_payout=payout, stake=stake, n_legs=n_legs)
    if result is None:
        print("Could not build a parlay from the available matches.")
        return

    print(f"\n{len(result.legs)}-leg parlay (target {payout:.1f}x):\n")
    for i, leg in enumerate(result.legs, start=1):
        edge = f"{leg.edge*100:+.1f}% edge" if leg.edge is not None else "no market data"
        print(f"  {i}. {leg.pick_label:35s} {leg.odds:6.2f}x   {edge}   ({leg.date})")
    print(f"\nCombined odds: {result.combined_odds:.2f}x (target was {payout:.1f}x)")
    print(f"Model's estimated chance all legs hit: {result.combined_model_prob*100:.1f}%")
    print(f"Stake ${result.stake:.2f} -> payout ${result.payout:.2f} (profit ${result.profit:.2f})")
    if result.used_non_edge_legs:
        print("\nNote: not enough positive-edge legs were available in range; some picks here have no model edge over the market.")
    print("\nNot a guarantee. For analysis/entertainment only.")

    if args.out:
        render_parlay_card(result, save_path=args.out)
        print(f"\nSaved card to {args.out}")


def cmd_refresh(args: argparse.Namespace) -> None:
    from wcwinner.pipeline import run_full_refresh

    run_full_refresh()


def main() -> None:
    parser = argparse.ArgumentParser(prog="wcwinner", description="World Cup 2026 Dixon-Coles prediction system")
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Predict a single matchup")
    p_predict.add_argument("home")
    p_predict.add_argument("away")
    p_predict.add_argument("--home-advantage", action="store_true", help="Non-neutral fixture with `home` at home")
    p_predict.add_argument("--knockout", action="store_true", help="Include ET/penalty advancement probability")
    p_predict.add_argument(
        "--home-strength", type=float, default=1.0,
        help="Manual squad-news multiplier on home xG, e.g. 0.85 for a missing striker (default 1.0, no adjustment)",
    )
    p_predict.add_argument(
        "--away-strength", type=float, default=1.0,
        help="Manual squad-news multiplier on away xG (default 1.0, no adjustment)",
    )
    p_predict.set_defaults(func=cmd_predict)

    p_bracket = sub.add_parser("bracket", help="Monte Carlo simulate the rest of the WC 2026 bracket")
    p_bracket.add_argument("--iterations", type=int, default=10000)
    p_bracket.add_argument("--seed", type=int, default=None)
    p_bracket.add_argument("--top", type=int, default=20)
    p_bracket.set_defaults(func=cmd_bracket)

    p_backtest = sub.add_parser("backtest", help="Backtest against held-out real matches")
    p_backtest.add_argument("--cutoff", default="2024-07-01")
    p_backtest.add_argument("--max-test-matches", type=int, default=None)
    p_backtest.set_defaults(func=cmd_backtest)

    p_market = sub.add_parser("market", help="Compare model probabilities to live betting odds")
    p_market.set_defaults(func=cmd_market)

    p_refresh = sub.add_parser("refresh", help="Re-download historical data and refit the model")
    p_refresh.set_defaults(func=cmd_refresh)

    p_card = sub.add_parser("card", help="Render a graphic prediction card to a PNG file")
    p_card.add_argument("home")
    p_card.add_argument("away")
    p_card.add_argument("--home-advantage", action="store_true")
    p_card.add_argument("--knockout", action="store_true")
    p_card.add_argument("--stage", default="ROUND OF 32")
    p_card.add_argument("--out", default="prediction_card.png")
    p_card.set_defaults(func=cmd_card)

    p_bet = sub.add_parser("betbuilder", help="Build a parlay targeting a payout multiple (asks interactively if flags are omitted)")
    p_bet.add_argument("--payout", type=float, default=None, help="Target payout multiple, e.g. 5 for 5x")
    p_bet.add_argument("--legs", type=int, default=None, help="Exact number of legs (default: let it choose)")
    p_bet.add_argument("--stake", type=float, default=None, help="Stake amount in dollars (default: 10)")
    p_bet.add_argument("--date-from", default=None, help="YYYY-MM-DD")
    p_bet.add_argument("--date-to", default=None, help="YYYY-MM-DD")
    p_bet.add_argument("--no-prompt-dates", action="store_true", help="Skip the date-range prompt and use all upcoming matches")
    p_bet.add_argument("--out", default=None, help="Optional PNG path to save a bet-slip card")
    p_bet.set_defaults(func=cmd_betbuilder)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
