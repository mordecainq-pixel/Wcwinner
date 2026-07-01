#!/usr/bin/env python3
"""Live bracket-state check. Pulls the current WC 2026 fixtures/standings
from football-data.org and validates the bracket-connectivity inference the
Monte Carlo simulator relies on for rounds the API hasn't resolved yet.

Run this before trusting `wcwinner bracket` output, especially right after a
knockout round finishes and the next one's pairings get revealed.

Usage: python scripts/check_bracket.py
"""
from collections import Counter

from wcwinner.data.football_data_client import bracket_state
from wcwinner.simulate.bracket import validate_connectivity


def main():
    bs = bracket_state()
    print(f"Current matchday: {bs['current_matchday']} (season {bs['season_start']} - {bs['season_end']})")

    counts = Counter((m["stage"], m["status"]) for m in bs["matches"])
    print("\nMatches by stage/status:")
    for (stage, status), n in sorted(counts.items()):
        print(f"  {stage:16s} {status:10s} {n}")

    warnings = validate_connectivity(bs["matches"])
    print()
    if warnings:
        print(
            "BRACKET CONNECTIVITY WARNINGS - the inferred pairing rule doesn't "
            "match a real resolved pairing. Do not trust `wcwinner bracket` "
            "results past the affected round until this is investigated:"
        )
        for w in warnings:
            print(" -", w)
    else:
        print("Bracket connectivity inference matches every currently-known pairing. No manual check needed yet.")


if __name__ == "__main__":
    main()
