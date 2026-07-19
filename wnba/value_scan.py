"""Admin-only: scan ONE sportsbook's live player-prop lines against the
model's own bootstrap projections and surface the biggest gaps as candidate
value bets -- e.g. the model projects a player's points at 17.5 but the book
has the line at 15.5, or the model's own probability at the book's exact
line clears a real positive-EV bar once the quoted odds are accounted for.

Costs real Odds-API credits per game scanned (same market as
betbuilder.py's player props -- see config.py's cost note), so this is
always an explicit, admin-triggered action. Results are cached to disk
(DATA/wnba_admin_cache/, gitignored and outside the daily-refresh commit
path -- see .gitignore and .github/workflows/wnba_daily_refresh.yml) so
reopening the admin panel just re-reads the last scan instead of spending
credits again, until you explicitly re-scan.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import pandas as pd

from wnba.betbuilder import out_player_ids_for_team
from wnba.config import DATA_PROCESSED_DIR
from wnba.data.odds_client import OddsAPIClient, player_milestone_odds, player_prop_lines
from wnba.model.player_model import prob_double_double, prob_over, prob_triple_double, simulate_player_games

VALUE_SCAN_CACHE_DIR = DATA_PROCESSED_DIR.parent / "wnba_admin_cache"
VALUE_SCAN_CACHE_FILE = VALUE_SCAN_CACHE_DIR / "value_scan.json"


@dataclass
class ValueRow:
    home_team: str
    away_team: str
    date: str
    player_name: str
    stat: str
    market_type: str  # "prop" or "milestone"
    book_line: float | None  # None for milestones -- no line, just a Yes price
    model_mean: float | None  # None for milestones
    gap: float | None  # model_mean - book_line; the intuitive "outlier" signal
    model_prob_over: float  # for milestones this is P(yes)
    market_prob_over: float | None
    over_odds: float | None
    under_odds: float | None
    ev_over: float | None
    ev_under: float | None
    best_side: str  # "over" / "under" / "yes"
    best_ev: float


def _resolve_player_id(player_box: pd.DataFrame, name: str) -> str | None:
    matches = player_box[player_box["player_name"].str.lower() == name.lower()]
    if matches.empty:
        return None
    return matches.iloc[-1]["player_id"]


def _prop_rows(
    home, away, date, opponent_for_player, player_box, opponent_factors, prop_df, out_ids, availability,
) -> list[ValueRow]:
    rows = []
    for r in prop_df.itertuples(index=False):
        player_id = _resolve_player_id(player_box, r.player_name)
        if player_id is None:
            continue
        try:
            sim = simulate_player_games(
                player_box, player_id, opponent_for_player, opponent_factors,
                out_player_ids=out_ids, availability=availability,
            )
        except ValueError:
            continue
        p_over = prob_over(sim, r.stat, r.line)
        ev_over = p_over * r.over_odds - 1.0
        ev_under = (1.0 - p_over) * r.under_odds - 1.0
        best_side, best_ev = ("over", ev_over) if ev_over >= ev_under else ("under", ev_under)
        rows.append(ValueRow(
            home_team=home, away_team=away, date=date, player_name=r.player_name, stat=r.stat,
            market_type="prop", book_line=r.line, model_mean=float(sim[r.stat].mean()),
            gap=float(sim[r.stat].mean()) - r.line, model_prob_over=p_over,
            market_prob_over=r.market_over_prob, over_odds=r.over_odds, under_odds=r.under_odds,
            ev_over=ev_over, ev_under=ev_under, best_side=best_side, best_ev=best_ev,
        ))
    return rows


def _milestone_rows(
    home, away, date, opponent_for_player, player_box, opponent_factors, milestone_df, out_ids, availability,
) -> list[ValueRow]:
    rows = []
    for r in milestone_df.itertuples(index=False):
        player_id = _resolve_player_id(player_box, r.player_name)
        if player_id is None:
            continue
        try:
            sim = simulate_player_games(
                player_box, player_id, opponent_for_player, opponent_factors,
                out_player_ids=out_ids, availability=availability,
            )
        except ValueError:
            continue
        model_prob = prob_double_double(sim) if r.stat == "double_double" else prob_triple_double(sim)
        ev_yes = model_prob * r.yes_odds - 1.0
        rows.append(ValueRow(
            home_team=home, away_team=away, date=date, player_name=r.player_name, stat=r.stat,
            market_type="milestone", book_line=None, model_mean=None, gap=None,
            model_prob_over=model_prob, market_prob_over=r.market_yes_prob_raw,
            over_odds=r.yes_odds, under_odds=None, ev_over=ev_yes, ev_under=None,
            best_side="yes", best_ev=ev_yes,
        ))
    return rows


def scan_bookmaker_value(
    matches: list,
    player_box: pd.DataFrame,
    opponent_factors: pd.DataFrame,
    bookmaker: str,
    availability: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    client: OddsAPIClient | None = None,
) -> list[ValueRow]:
    """Scans the given MatchOptions (from betbuilder.gather_match_options,
    must have event_id set) for ONE bookmaker's player-prop + milestone
    lines, comparing each to the model's own projection. This is the
    expensive call (real Odds-API credits, one event-props request per
    match) -- only invoke it for matches you've actually selected, and cache
    the result (see save_scan/load_cached_scan) rather than re-calling it on
    every page load.
    """
    client = client or OddsAPIClient()
    rows: list[ValueRow] = []
    for match in matches:
        if match.event_id is None:
            continue
        home, away, date = match.home_team, match.away_team, match.date
        try:
            prop_df = player_prop_lines(match.event_id, client=client, bookmakers=bookmaker)
            milestone_df = player_milestone_odds(match.event_id, client=client, bookmakers=bookmaker)
        except Exception:
            continue

        for team, opponent in ((home, away), (away, home)):
            team_players = player_box[player_box["team"] == team]["player_name"].unique()
            out_ids = out_player_ids_for_team(injuries, opponent) if injuries is not None else None
            if not prop_df.empty:
                team_prop_df = prop_df[prop_df["player_name"].isin(team_players)]
                rows += _prop_rows(home, away, date, opponent, player_box, opponent_factors, team_prop_df, out_ids, availability)
            if not milestone_df.empty:
                team_milestone_df = milestone_df[milestone_df["player_name"].isin(team_players)]
                rows += _milestone_rows(home, away, date, opponent, player_box, opponent_factors, team_milestone_df, out_ids, availability)

    return sorted(rows, key=lambda r: r.best_ev, reverse=True)


def save_scan(rows: list[ValueRow], bookmaker: str) -> None:
    VALUE_SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "bookmaker": bookmaker,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rows": [asdict(r) for r in rows],
    }
    VALUE_SCAN_CACHE_FILE.write_text(json.dumps(payload, indent=2))


def load_cached_scan() -> dict | None:
    """Returns {"bookmaker", "fetched_at", "rows": [...]} from the last
    scan, or None if a scan has never been run (or the cache dir was wiped
    -- e.g. a fresh deploy, since this directory is gitignored on purpose).
    """
    if not VALUE_SCAN_CACHE_FILE.exists():
        return None
    return json.loads(VALUE_SCAN_CACHE_FILE.read_text())
