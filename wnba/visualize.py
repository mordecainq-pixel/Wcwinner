"""Renders predictions and prop projections as graphic "cards" -- same
visual language as wcwinner/visualize.py (same palette, same card
anatomy) so the two products feel like one system, adapted for
basketball's two-way (no draw) outcome and for player props, which
soccer doesn't have.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from wnba.model.margin_model import MarginModel
from wnba.model.player_model import STAT_LABELS

HOME_COLOR = "#3c8c40"
AWAY_COLOR = "#e0952b"
BG_COLOR = "#f2efe9"
PANEL_COLOR = "#e9e5db"


def _scaled_ratings(model: MarginModel) -> dict[str, dict[str, float]]:
    teams = list(model.attack.keys())
    attack_vals = np.array([model.attack[t] for t in teams])
    defense_vals = np.array([-model.defense[t] for t in teams])  # sign-flipped: higher = better defense

    def scale(vals: np.ndarray) -> np.ndarray:
        lo, hi = vals.min(), vals.max()
        return 30 + 65 * (vals - lo) / (hi - lo) if hi > lo else np.full_like(vals, 50.0)

    attack_scaled = scale(attack_vals)
    defense_scaled = scale(defense_vals)
    overall = (attack_scaled + defense_scaled) / 2
    return {t: {"attack": float(a), "defense": float(d), "overall": float(o)} for t, a, d, o in zip(teams, attack_scaled, defense_scaled, overall)}


def render_prediction_card(
    model: MarginModel,
    home_team: str,
    away_team: str,
    neutral: bool = False,
    market_row: dict | None = None,
    save_path: str | None = None,
):
    """`market_row` (optional): a dict-like from odds_client.game_market_probabilities
    for this exact matchup, e.g. {"market_home_prob": 0.6, "market_away_prob": 0.4}.
    """
    win_prob = model.win_probability(home_team, away_team, neutral=neutral)
    pred_home, pred_away = model.predicted_scores(home_team, away_team, neutral=neutral)
    total_mu, _ = model.total_distribution(home_team, away_team, neutral=neutral)
    ratings = _scaled_ratings(model)
    home_r = ratings.get(home_team, {"attack": 50.0, "defense": 50.0})
    away_r = ratings.get(away_team, {"attack": 50.0, "defense": 50.0})

    has_market = market_row is not None and market_row.get("market_home_prob") is not None
    top = 10.6 if has_market else 8.7
    fig, ax = plt.subplots(figsize=(6.4, top * 0.62), facecolor=BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, top)
    ax.axis("off")

    def name_fontsize(name: str) -> float:
        return max(8.0, min(13.0, 150.0 / max(len(name), 1)))

    ax.text(5, top - 0.4, "WNBA · MARGIN/TOTAL MODEL", ha="center", fontsize=11, weight="bold", color="#555")
    ax.text(1.9, top - 1.3, home_team.upper(), ha="center", fontsize=name_fontsize(home_team), weight="bold")
    ax.text(8.1, top - 1.3, away_team.upper(), ha="center", fontsize=name_fontsize(away_team), weight="bold")
    ax.text(5, top - 1.3, "vs", ha="center", fontsize=9, color="#aaa")

    ax.text(5, top - 2.15, "WIN PROBABILITY", ha="center", fontsize=10, color="#666")
    x0, width, bar_h = 0.5, 9.0, 0.55
    bar_y = top - 2.9
    home_w = width * win_prob
    ax.add_patch(mpatches.Rectangle((x0, bar_y), home_w, bar_h, color=HOME_COLOR))
    ax.add_patch(mpatches.Rectangle((x0 + home_w, bar_y), width - home_w, bar_h, color=AWAY_COLOR))
    ax.text(x0 + home_w / 2, bar_y + bar_h / 2, f"{win_prob*100:.0f}%", ha="center", va="center", fontsize=12, weight="bold", color="white")
    ax.text(x0 + home_w + (width - home_w) / 2, bar_y + bar_h / 2, f"{(1-win_prob)*100:.0f}%", ha="center", va="center", fontsize=12, weight="bold", color="white")

    y = bar_y - 0.95
    for label, hv, av in [
        ("ATTACK SCORE", f"{home_r['attack']:.0f}", f"{away_r['attack']:.0f}"),
        ("DEFENSE SCORE", f"{home_r['defense']:.0f}", f"{away_r['defense']:.0f}"),
        ("PROJECTED SCORE", f"{pred_home:.1f}", f"{pred_away:.1f}"),
    ]:
        ax.text(2, y, hv, ha="center", fontsize=16, weight="bold", color=HOME_COLOR)
        ax.text(5, y, label, ha="center", fontsize=9, color="#888")
        ax.text(8, y, av, ha="center", fontsize=16, weight="bold", color=AWAY_COLOR)
        y -= 0.95

    if has_market:
        market_home = market_row["market_home_prob"]
        flip = (market_home > 0.5) != (win_prob > 0.5)
        ax.text(2, y, f"{market_home*100:.0f}%", ha="center", fontsize=14, weight="bold", color=HOME_COLOR)
        ax.text(5, y, "MARKET WIN" + ("\n— flip —" if flip else ""), ha="center", fontsize=9, color="#888")
        ax.text(8, y, f"{(1-market_home)*100:.0f}%", ha="center", fontsize=14, weight="bold", color=AWAY_COLOR)
        y -= 0.95

    ax.add_patch(mpatches.FancyBboxPatch((1.2, y - 1.0), 7.6, 0.95, boxstyle="round,pad=0.02", linewidth=0, facecolor=PANEL_COLOR))
    ax.text(1.7, y - 0.52, "Projected total", ha="left", fontsize=10, color="#555")
    ax.text(8.3, y - 0.52, f"{total_mu:.1f}", ha="right", fontsize=15, weight="bold")
    y -= 1.35

    lean_team = home_team if win_prob > 0.5 else away_team
    ax.text(5, max(y, 0.3), f"Model lean: {lean_team}", ha="center", fontsize=10, style="italic", color="#444")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def render_prop_card(player_name: str, opponent: str, sim, primary_stat: str, line: float | None = None, save_path: str | None = None):
    """`sim`: a simulate_player_games() DataFrame. Shows the primary stat's
    projected distribution plus a compact table of every other projected
    stat/combo.
    """
    from wnba.model.player_model import prob_double_double, prob_over, prob_triple_double

    mean, median, std = sim[primary_stat].mean(), sim[primary_stat].median(), sim[primary_stat].std()
    other_stats = [c for c in ("points", "rebounds", "assists", "fg3m", "steals", "blocks", "turnovers", "pra") if c != primary_stat]

    top = 11.6 if line is not None else 10.6
    fig, ax = plt.subplots(figsize=(6.4, top * 0.62), facecolor=BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, top)
    ax.axis("off")

    ax.text(5, top - 0.5, "WNBA · PLAYER PROP PROJECTION", ha="center", fontsize=11, weight="bold", color="#555")
    ax.text(5, top - 1.4, player_name.upper(), ha="center", fontsize=17, weight="bold")
    ax.text(5, top - 2.0, f"vs {opponent}", ha="center", fontsize=10, color="#888")

    stat_label = STAT_LABELS.get(primary_stat, primary_stat)
    ax.add_patch(mpatches.FancyBboxPatch((0.6, top - 3.9), 8.8, 1.55, boxstyle="round,pad=0.03", linewidth=0, facecolor=PANEL_COLOR))
    ax.text(5, top - 2.75, stat_label.upper(), ha="center", fontsize=10, color="#666")
    ax.text(5, top - 3.55, f"{mean:.1f}", ha="center", fontsize=30, weight="bold", color=HOME_COLOR)
    ax.text(1.5, top - 3.55, f"median\n{median:.1f}", ha="center", fontsize=9, color="#777")
    ax.text(8.5, top - 3.55, f"std dev\n{std:.1f}", ha="center", fontsize=9, color="#777")

    y = top - 4.5
    if line is not None:
        p_over = prob_over(sim, primary_stat, line)
        x0, width, bar_h = 0.7, 8.6, 0.55
        over_w = width * p_over
        ax.add_patch(mpatches.Rectangle((x0, y), over_w, bar_h, color=HOME_COLOR))
        ax.add_patch(mpatches.Rectangle((x0 + over_w, y), width - over_w, bar_h, color=AWAY_COLOR))
        ax.text(x0 + over_w / 2, y + bar_h / 2, f"Over {line}\n{p_over*100:.0f}%", ha="center", va="center", fontsize=9.5, weight="bold", color="white")
        ax.text(x0 + over_w + (width - over_w) / 2, y + bar_h / 2, f"Under {line}\n{(1-p_over)*100:.0f}%", ha="center", va="center", fontsize=9.5, weight="bold", color="white")
        y -= 1.0

    y -= 0.3
    ax.text(0.7, y, "OTHER PROJECTIONS", ha="left", fontsize=9, color="#888")
    y -= 0.5
    for stat in other_stats[:6]:
        label = STAT_LABELS.get(stat, stat)
        ax.text(0.9, y, label, ha="left", fontsize=10, color="#444")
        ax.text(9.1, y, f"{sim[stat].mean():.1f}", ha="right", fontsize=10, weight="bold")
        y -= 0.5

    y -= 0.2
    ax.text(0.9, y, "Double-Double", ha="left", fontsize=10, color="#444")
    ax.text(9.1, y, f"{prob_double_double(sim)*100:.0f}%", ha="right", fontsize=10, weight="bold")
    y -= 0.5
    ax.text(0.9, y, "Triple-Double", ha="left", fontsize=10, color="#444")
    ax.text(9.1, y, f"{prob_triple_double(sim)*100:.0f}%", ha="right", fontsize=10, weight="bold")
    y -= 0.65

    ax.text(5, max(y, 0.3), "Model projection only, not a guarantee.", ha="center", fontsize=8, color="#999")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def render_parlay_card(result, save_path: str | None = None):
    """Renders a `betbuilder.ParlayResult` (game and/or player-prop legs)
    as a bet-slip style card. Structurally identical to wcwinner's version
    -- Leg.market_type/pick_label/reasoning are already sport-appropriate
    text by the time they reach here, so nothing basketball-specific is
    needed in the rendering itself.
    """
    n = len(result.legs)
    top = 1.9 + 0.95 * n + 0.15 + 1.2 + 0.9 + 0.45
    fig, ax = plt.subplots(figsize=(6.4, top * 0.68), facecolor=BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, top)
    ax.axis("off")

    ax.text(5, top - 0.5, "WNBA BET BUILDER", ha="center", fontsize=11, weight="bold", color="#555")
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
    ax.add_patch(mpatches.FancyBboxPatch((0.4, y - 1.2), 9.2, 1.15, boxstyle="round,pad=0.02", linewidth=0, facecolor="#333"))
    ax.text(0.7, y - 0.35, "Combined odds", ha="left", fontsize=10, color="#ccc")
    ax.text(9.1, y - 0.35, f"{result.combined_odds:.2f}x", ha="right", fontsize=12, weight="bold", color="white")
    ax.text(0.7, y - 0.8, f"Stake ${result.stake:.2f}  →  Payout", ha="left", fontsize=10, color="#ccc")
    ax.text(9.1, y - 0.8, f"${result.payout:.2f}", ha="right", fontsize=13, weight="bold", color="#7bc47f")
    y -= 1.55

    ax.text(
        5, max(y, 0.9),
        f"Model's estimated chance ALL {n} legs hit: {result.combined_model_prob*100:.1f}%",
        ha="center", fontsize=9.5, style="italic", color="#555",
    )
    ax.text(5, 0.25, "Not a guarantee. For analysis/entertainment only - bet responsibly.", ha="center", fontsize=8, color="#999")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
