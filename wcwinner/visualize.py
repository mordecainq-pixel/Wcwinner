"""Renders a single-matchup prediction as a graphic "card" (team ratings,
win-probability bar, expected goals, market comparison, knockout advance %,
projected score) rather than a plain-text table.

Team flags aren't drawn (no flag image assets in this project) -- team names
are shown as text instead. Attack/Defense "scores" are cosmetic 30-95 scaled
versions of the fitted Dixon-Coles parameters (min-max normalized across all
fitted teams), for readability only -- they are not new model outputs.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from wcwinner.data.odds_client import market_probabilities
from wcwinner.model.dixon_coles import DixonColesModel
from wcwinner.simulate.match import predict_match

HOME_COLOR = "#3c8c40"
AWAY_COLOR = "#e0952b"
DRAW_COLOR = "#b0b0b0"
BG_COLOR = "#f2efe9"
PANEL_COLOR = "#e9e5db"


def _scaled_ratings(model: DixonColesModel) -> dict[str, dict[str, float]]:
    teams = list(model.attack.keys())
    attack_vals = np.array([model.attack[t] for t in teams])
    defense_vals = np.array([-model.defense[t] for t in teams])  # sign-flipped: higher = better defense

    def scale(vals: np.ndarray) -> np.ndarray:
        lo, hi = vals.min(), vals.max()
        return 30 + 65 * (vals - lo) / (hi - lo)

    attack_scaled = scale(attack_vals)
    defense_scaled = scale(defense_vals)
    overall = (attack_scaled + defense_scaled) / 2
    return {
        t: {"attack": float(a), "defense": float(d), "overall": float(o)}
        for t, a, d, o in zip(teams, attack_scaled, defense_scaled, overall)
    }


def _market_row_for(home_team: str, away_team: str, market_df) -> dict | None:
    if market_df is None or market_df.empty:
        return None
    match = market_df[(market_df["home_team"] == home_team) & (market_df["away_team"] == away_team)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def render_prediction_card(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    elo_ratings: dict[str, float],
    neutral: bool = True,
    knockout: bool = False,
    stage_label: str = "ROUND OF 32",
    include_market: bool = True,
    save_path: str | None = None,
):
    pred = predict_match(model, home_team, away_team, neutral=neutral, knockout=knockout, elo_ratings=elo_ratings)
    ratings = _scaled_ratings(model)
    home_r = ratings.get(home_team, {"attack": 50.0, "defense": 50.0, "overall": 50.0})
    away_r = ratings.get(away_team, {"attack": 50.0, "defense": 50.0, "overall": 50.0})

    market_row = None
    if include_market:
        try:
            market_row = _market_row_for(home_team, away_team, market_probabilities())
        except Exception:
            market_row = None  # market data is a nice-to-have on the card, never required

    fig, ax = plt.subplots(figsize=(6.4, 8.5), facecolor=BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    def name_fontsize(name: str) -> float:
        return max(8.0, min(13.0, 150.0 / max(len(name), 1)))

    ax.text(5, 13.5, f"{stage_label} · DIXON-COLES MODEL", ha="center", fontsize=11, weight="bold", color="#555")
    ax.text(1.9, 12.6, home_team.upper(), ha="center", fontsize=name_fontsize(home_team), weight="bold")
    ax.text(8.1, 12.6, away_team.upper(), ha="center", fontsize=name_fontsize(away_team), weight="bold")
    ax.text(5, 12.6, "vs", ha="center", fontsize=9, color="#aaa")

    ax.text(5, 11.75, "WIN IN REGULATION", ha="center", fontsize=10, color="#666")
    p_home, p_draw, p_away = pred["p_home_win"], pred["p_draw"], pred["p_away_win"]
    x0, width, bar_y, bar_h = 0.5, 9.0, 11.0, 0.55
    home_w, draw_w, away_w = width * p_home, width * p_draw, width * p_away
    ax.add_patch(mpatches.Rectangle((x0, bar_y), home_w, bar_h, color=HOME_COLOR))
    ax.add_patch(mpatches.Rectangle((x0 + home_w, bar_y), draw_w, bar_h, color=DRAW_COLOR))
    ax.add_patch(mpatches.Rectangle((x0 + home_w + draw_w, bar_y), away_w, bar_h, color=AWAY_COLOR))
    ax.text(x0 + home_w / 2, bar_y + bar_h / 2, f"{p_home*100:.0f}%", ha="center", va="center", fontsize=11, weight="bold", color="white")
    ax.text(x0 + home_w + draw_w / 2, bar_y + bar_h / 2, f"{p_draw*100:.0f}%", ha="center", va="center", fontsize=11, weight="bold", color="#444")
    ax.text(x0 + home_w + draw_w + away_w / 2, bar_y + bar_h / 2, f"{p_away*100:.0f}%", ha="center", va="center", fontsize=11, weight="bold", color="white")
    ax.text(2, bar_y - 0.35, home_team, ha="center", fontsize=9, color=HOME_COLOR, weight="bold")
    ax.text(5, bar_y - 0.35, "Draw", ha="center", fontsize=9, color="#888")
    ax.text(8, bar_y - 0.35, away_team, ha="center", fontsize=9, color=AWAY_COLOR, weight="bold")

    y = 9.7
    for label, hv, av in [
        ("ATTACK SCORE", f"{home_r['attack']:.0f}", f"{away_r['attack']:.0f}"),
        ("DEFENSE SCORE", f"{home_r['defense']:.0f}", f"{away_r['defense']:.0f}"),
        ("EXPECTED GOALS", f"{pred['expected_goals_home']:.2f}", f"{pred['expected_goals_away']:.2f}"),
    ]:
        ax.text(2, y, hv, ha="center", fontsize=16, weight="bold", color=HOME_COLOR)
        ax.text(5, y, label, ha="center", fontsize=9, color="#888")
        ax.text(8, y, av, ha="center", fontsize=16, weight="bold", color=AWAY_COLOR)
        y -= 0.95

    if market_row is not None:
        market_home, market_away = market_row["market_home_prob"], market_row["market_away_prob"]
        flip = (market_home > market_away) != (p_home > p_away)
        ax.text(2, y, f"{market_home*100:.0f}%", ha="center", fontsize=14, weight="bold", color=HOME_COLOR)
        ax.text(5, y, "MARKET WIN" + ("\n— flip —" if flip else ""), ha="center", fontsize=9, color="#888")
        ax.text(8, y, f"{market_away*100:.0f}%", ha="center", fontsize=14, weight="bold", color=AWAY_COLOR)
        y -= 0.95

    if knockout:
        ko = pred["knockout"]
        ax.text(2, y, f"{ko['p_home_advances']*100:.0f}%", ha="center", fontsize=16, weight="bold", color=HOME_COLOR)
        ax.text(5, y, "TO ADVANCE", ha="center", fontsize=9, color="#888")
        ax.text(8, y, f"{ko['p_away_advances']*100:.0f}%", ha="center", fontsize=16, weight="bold", color=AWAY_COLOR)
        y -= 1.05

    i, j = pred["most_likely_score"]
    ax.add_patch(mpatches.FancyBboxPatch((1.2, y - 1.55), 7.6, 1.45, boxstyle="round,pad=0.02", linewidth=0, facecolor=PANEL_COLOR))
    ax.text(1.7, y - 0.55, "Projected score", ha="left", fontsize=10, color="#555")
    ax.text(8.3, y - 0.55, f"{i} - {j}", ha="right", fontsize=15, weight="bold")
    ax.text(1.7, y - 1.15, "Both teams to score", ha="left", fontsize=10, color="#555")
    ax.text(8.3, y - 1.15, f"{pred['p_btts']*100:.0f}%", ha="right", fontsize=12, weight="bold")
    y -= 1.9

    lean_team = home_team if p_home > p_away else away_team
    lean_note = ""
    if market_row is not None:
        market_favorite = home_team if market_home > market_away else away_team
        if market_favorite != lean_team:
            lean_note = f" — market likes {market_favorite}, model flips it"
    ax.text(5, max(y, 0.4), f"Model lean: {lean_team}{lean_note}", ha="center", fontsize=10, style="italic", color="#444")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def render_parlay_card(result, save_path: str | None = None):
    """Renders a `betbuilder.ParlayResult` as a bet-slip style card."""
    from wcwinner.betbuilder import ParlayResult  # local import to avoid a cycle at module load

    assert isinstance(result, ParlayResult)
    n = len(result.legs)
    top = 1.9 + 0.95 * n + 0.15 + 1.55 + 0.9 + (0.4 if result.used_non_edge_legs else 0) + 0.45
    fig, ax = plt.subplots(figsize=(6.4, top * 0.68), facecolor=BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, top)
    ax.axis("off")

    ax.text(5, top - 0.5, "BET BUILDER · DIXON-COLES MODEL", ha="center", fontsize=11, weight="bold", color="#555")
    ax.text(5, top - 1.15, f"{n}-Leg Parlay", ha="center", fontsize=15, weight="bold")

    y = top - 1.9
    for i, leg in enumerate(result.legs, start=1):
        edge_color = "#888"
        edge_text = "no market data"
        if leg.edge is not None:
            edge_color = HOME_COLOR if leg.ev > 0 else AWAY_COLOR
            edge_text = f"{leg.ev*100:+.1f}% EV"

        ax.add_patch(mpatches.FancyBboxPatch((0.4, y - 0.68), 9.2, 0.72, boxstyle="round,pad=0.02", linewidth=0, facecolor=PANEL_COLOR))
        ax.text(0.65, y - 0.22, f"{i}. [{leg.market_type}] {leg.pick_label}", ha="left", fontsize=10.5, weight="bold")
        ax.text(0.65, y - 0.55, f"{leg.home_team} vs {leg.away_team}  ·  {leg.date}", ha="left", fontsize=8.5, color="#777")
        ax.text(9.35, y - 0.22, f"{leg.odds:.2f}x", ha="right", fontsize=11, weight="bold", color=HOME_COLOR)
        ax.text(9.35, y - 0.55, edge_text, ha="right", fontsize=8.5, color=edge_color, weight="bold")
        y -= 0.95

    y -= 0.15
    ax.add_patch(mpatches.FancyBboxPatch((0.4, y - 1.55), 9.2, 1.5, boxstyle="round,pad=0.02", linewidth=0, facecolor="#333"))
    ax.text(0.7, y - 0.35, "Target payout", ha="left", fontsize=10, color="#ccc")
    ax.text(9.1, y - 0.35, f"{result.target_payout:.1f}x", ha="right", fontsize=12, weight="bold", color="white")
    ax.text(0.7, y - 0.75, "Actual combined odds", ha="left", fontsize=10, color="#ccc")
    ax.text(9.1, y - 0.75, f"{result.combined_odds:.2f}x", ha="right", fontsize=12, weight="bold", color="white")
    ax.text(0.7, y - 1.15, f"Stake ${result.stake:.2f}  →  Payout", ha="left", fontsize=10, color="#ccc")
    ax.text(9.1, y - 1.15, f"${result.payout:.2f}", ha="right", fontsize=13, weight="bold", color="#7bc47f")
    y -= 1.9

    ax.text(
        5, max(y, 0.9),
        f"Model's estimated chance ALL {n} legs hit: {result.combined_model_prob*100:.1f}%",
        ha="center", fontsize=9.5, style="italic", color="#555",
    )
    if result.used_non_edge_legs:
        ax.text(5, max(y - 0.4, 0.55), "Note: not enough positive-edge legs available in range - some picks here have no model edge over the market.", ha="center", fontsize=7.5, color="#b05c1a", wrap=True)
    ax.text(5, 0.25, "Not a guarantee. For analysis/entertainment only - bet responsibly.", ha="center", fontsize=8, color="#999")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def render_tournament_odds_card(sim_result, n_iterations: int, top_n: int = 12, save_path: str | None = None):
    """Horizontal bar chart of championship odds from `simulate.bracket.simulate_tournament`."""
    top = sim_result.head(top_n).iloc[::-1]  # reverse so the favorite ends up on top of the chart

    fig, ax = plt.subplots(figsize=(6.4, 0.55 * top_n + 1.6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bars = ax.barh(top.index, top["CHAMPION"] * 100, color=HOME_COLOR, height=0.6)
    for bar, (team, row) in zip(bars, top.iterrows()):
        ax.text(
            bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
            f"{row['CHAMPION']*100:.1f}%", va="center", ha="left", fontsize=10, weight="bold", color="#333",
        )

    ax.set_xlim(0, max(top["CHAMPION"] * 100) * 1.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=10, weight="bold")
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Chance of winning the tournament (%)", fontsize=9, color="#666")
    ax.set_title(f"WORLD CUP 2026 · MONTE CARLO SIMULATION ({n_iterations:,} iterations)", fontsize=11, weight="bold", color="#555", pad=14)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
