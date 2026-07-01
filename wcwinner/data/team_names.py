"""Aligns team names across data sources onto the Kaggle dataset's naming convention.

Kaggle's ``results.csv`` naming is treated as canonical since it's what the Elo
ratings and Dixon-Coles model are fit against. football-data.org, API-Football,
and The Odds API each spell a handful of teams differently, which breaks joins
if left alone.
"""
from __future__ import annotations

import difflib
import unicodedata

# Known spelling variants -> Kaggle canonical name. Covers mismatches actually
# observed across football-data.org, API-Football, and The Odds API for the
# 2026 World Cup field; extend as new mismatches surface.
ALIASES: dict[str, str] = {
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegowina": "Bosnia and Herzegovina",
    "cape verde islands": "Cape Verde",
    "cabo verde": "Cape Verde",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "congo democratic republic": "DR Congo",
    "czechia": "Czech Republic",
    "usa": "United States",
    "us": "United States",
    "united states of america": "United States",
    "ivory coast": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "korea south": "South Korea",
    "ir iran": "Iran",
    "islamic republic of iran": "Iran",
    "usvi": "United States Virgin Islands",
    "st kitts and nevis": "Saint Kitts and Nevis",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
}


def _normalize_key(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.strip().lower()


def to_canonical(name: str, known_teams: list[str] | None = None) -> str:
    """Map a team name from any supported source onto the Kaggle canonical spelling.

    Falls back to fuzzy matching against ``known_teams`` (e.g. the set of team
    names seen in the Kaggle dataset) when there's no exact alias, and returns
    the input unchanged if nothing matches closely enough.
    """
    key = _normalize_key(name)
    if key in ALIASES:
        return ALIASES[key]

    if known_teams:
        canonical_by_key = {_normalize_key(t): t for t in known_teams}
        if key in canonical_by_key:
            return canonical_by_key[key]
        close = difflib.get_close_matches(key, canonical_by_key.keys(), n=1, cutoff=0.85)
        if close:
            return canonical_by_key[close[0]]

    return name
