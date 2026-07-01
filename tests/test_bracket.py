from wcwinner.simulate.bracket import infer_next_round_pairs, validate_connectivity


def _match(match_id, home, away, date, status="FINISHED", home_score=None, away_score=None, stage="LAST_32"):
    return {
        "match_id": match_id,
        "utc_date": date,
        "status": status,
        "stage": stage,
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
    }


def test_infer_next_round_pairs_mirrors_top_and_bottom_half():
    matches = [_match(i, f"H{i}", f"A{i}", f"2026-07-0{i}") for i in range(1, 7)]
    pairs = infer_next_round_pairs(matches)
    assert len(pairs) == 3
    assert pairs[0] == (matches[0], matches[3])
    assert pairs[1] == (matches[1], matches[4])
    assert pairs[2] == (matches[2], matches[5])


def test_validate_connectivity_passes_when_inferred_pairing_matches_known_result():
    r32 = [
        _match(1, "A", "B", "2026-06-28", home_score=2, away_score=0),
        _match(2, "C", "D", "2026-06-29", home_score=1, away_score=0),
        _match(3, "E", "F", "2026-06-29", home_score=0, away_score=1),
        _match(4, "G", "H", "2026-06-30", home_score=2, away_score=1),
    ]
    # A beats B, C beats D, F beats E, G beats H.
    # infer_next_round_pairs pairs (1,3) and (2,4) -> winners {A,F} and {C,G}.
    r16 = [
        _match(5, "A", "F", "2026-07-02", status="TIMED", stage="LAST_16"),
        _match(6, "C", "G", "2026-07-03", status="TIMED", stage="LAST_16"),
    ]
    warnings = validate_connectivity(r32 + r16)
    assert warnings == []


def test_validate_connectivity_flags_mismatch():
    r32 = [
        _match(1, "A", "B", "2026-06-28", home_score=2, away_score=0),
        _match(2, "C", "D", "2026-06-29", home_score=1, away_score=0),
        _match(3, "E", "F", "2026-06-29", home_score=0, away_score=1),
        _match(4, "G", "H", "2026-06-30", home_score=2, away_score=1),
    ]
    # Deliberately wrong known pairing: A vs C instead of the actual A vs F.
    r16 = [
        _match(5, "A", "C", "2026-07-02", status="TIMED", stage="LAST_16"),
    ]
    warnings = validate_connectivity(r32 + r16)
    assert len(warnings) > 0
