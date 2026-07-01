"""Monte Carlo knockout bracket simulation.

football-data.org's free tier gives us every already-determined fixture and
leaves undetermined future slots as None/None placeholders — it does not
expose the abstract bracket tree (which two slots feed a given future slot)
for rounds that haven't been reached yet. That connectivity is inferred here
using the standard "top half of the draw meets the mirrored bottom half"
single-elimination convention (pair match i with match i + n/2 within a
sorted-by-kickoff-time stage), then self-checked against whichever slots are
already concretely known.

As of the 2026 Round of 32, that self-check passes for the 3 Round-of-16
pairings football-data.org has already resolved (Canada v Morocco, Paraguay
v France, Brazil v Norway) — see `validate_connectivity`. Deeper rounds
(quarterfinals onward) use the same rule by extension but are NOT yet
independently verifiable against real data; treat far-round percentages as
best-effort until the API confirms more of the tree, per the project's
"flag what needs a live check" policy. `validate_connectivity` should be
re-run after each round completes.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from wcwinner.data.football_data_client import bracket_state
from wcwinner.model.dixon_coles import DixonColesModel
from wcwinner.model.knockout import sample_knockout_winner

STAGE_ORDER = ["LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]

# What winning a given stage's matches means for the winner.
ROUND_REACHED_BY_WINNING = {
    "LAST_32": "LAST_16",
    "LAST_16": "QUARTER_FINALS",
    "QUARTER_FINALS": "SEMI_FINALS",
    "SEMI_FINALS": "FINAL",
    "FINAL": "CHAMPION",
}


def _stage_matches(matches: list[dict], stage: str) -> list[dict]:
    return sorted(
        [m for m in matches if m["stage"] == stage],
        key=lambda m: m["utc_date"],
    )


def infer_next_round_pairs(current_stage_matches: list[dict]) -> list[tuple[dict, dict]]:
    n = len(current_stage_matches)
    half = n // 2
    return [(current_stage_matches[i], current_stage_matches[i + half]) for i in range(half)]


def validate_connectivity(matches: list[dict]) -> list[str]:
    """Check the inferred pairing rule against whichever next-round slots are
    already concretely known (both teams resolved). Returns a list of
    human-readable warnings; empty means everything checkable matches.
    """
    warnings = []
    for i, stage in enumerate(STAGE_ORDER[:-1]):
        next_stage = STAGE_ORDER[i + 1]
        cur = _stage_matches(matches, stage)
        nxt = _stage_matches(matches, next_stage)
        if not cur or not nxt:
            continue
        known_next_team_sets = [
            frozenset([m["home_team"], m["away_team"]])
            for m in nxt
            if m["home_team"] and m["away_team"]
        ]
        if not known_next_team_sets:
            continue

        pairs = infer_next_round_pairs(cur)
        for match_a, match_b in pairs:
            winner_a = _known_winner(match_a)
            winner_b = _known_winner(match_b)
            if winner_a is None or winner_b is None:
                continue  # can't check a pair whose feeder result isn't in yet
            inferred = frozenset([winner_a, winner_b])
            if inferred not in known_next_team_sets:
                warnings.append(
                    f"{stage}->{next_stage}: inferred pairing {set(inferred)} not found "
                    f"among known {next_stage} matchups {known_next_team_sets}"
                )
    return warnings


def _known_winner(match: dict) -> str | None:
    if match["status"] != "FINISHED" or match["home_score"] is None:
        return None
    return match["home_team"] if match["home_score"] > match["away_score"] else match["away_team"]


def simulate_tournament(
    model: DixonColesModel,
    elo_ratings: dict[str, float],
    bracket: dict | None = None,
    n_iterations: int = 10000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Monte Carlo the remaining bracket `n_iterations` times. Already-finished
    matches use their real result; everything else is sampled from the
    Dixon-Coles score distribution (with the knockout tiebreak for draws).

    Returns a DataFrame indexed by team with one column per stage: the
    fraction of iterations in which that team reached at least that stage
    (LAST_16 / QUARTER_FINALS / SEMI_FINALS / FINAL / CHAMPION).
    """
    bracket = bracket or bracket_state()
    matches = bracket["matches"]
    rng = np.random.default_rng(seed)

    r32 = _stage_matches(matches, "LAST_32")
    all_teams = sorted({m["home_team"] for m in r32} | {m["away_team"] for m in r32})

    reach_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    champion_counts: dict[str, int] = defaultdict(int)

    for _ in range(n_iterations):
        current_round_matches = r32

        for stage in STAGE_ORDER:
            reached_stage = ROUND_REACHED_BY_WINNING[stage]
            winners_by_id = {}
            for m in current_round_matches:
                known = _known_winner(m) if stage == "LAST_32" else None
                winner = known if known is not None else sample_knockout_winner(
                    model, m["home_team"], m["away_team"], elo_ratings, rng
                )
                winners_by_id[m["match_id"]] = winner
                reach_counts[winner][reached_stage] += 1

            if stage == "FINAL":
                champion_counts[next(iter(winners_by_id.values()))] += 1
                break

            pairs = infer_next_round_pairs(current_round_matches)
            current_round_matches = [
                {
                    "match_id": f"{stage}_next_{idx}",
                    "home_team": winners_by_id[match_a["match_id"]],
                    "away_team": winners_by_id[match_b["match_id"]],
                }
                for idx, (match_a, match_b) in enumerate(pairs)
            ]

    stages = ["LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]
    rows = []
    for team in all_teams:
        row = {"team": team}
        for stage in stages:
            row[stage] = reach_counts[team][stage] / n_iterations
        row["CHAMPION"] = champion_counts[team] / n_iterations
        rows.append(row)

    return pd.DataFrame(rows).set_index("team").sort_values("CHAMPION", ascending=False)
