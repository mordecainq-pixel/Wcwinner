import wcwinner.simulate.bracket as bracket_module
from wcwinner.model.dixon_coles import DixonColesModel
from wcwinner.simulate.bracket import infer_next_round_pairs, simulate_tournament, validate_connectivity


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


def test_simulate_tournament_prefers_real_pairing_over_inference(monkeypatch):
    """Regression test for a real bug found in production: the inferred
    mirrored-draw pairing rule happened to match the first 3 known Round of
    16 pairings by coincidence, then was proven wrong once all 8 were
    revealed. simulate_tournament must use the API's own confirmed next-round
    pairing whenever it's fully known, only falling back to inference for
    genuinely undetermined future rounds.

    Uses a full 32-team bracket (not a toy 8-team one) so it cascades
    correctly through every stage, same as the real tournament.
    """
    teams = [f"Team{i}" for i in range(1, 33)]
    model = DixonColesModel(
        attack={t: 0.3 for t in teams},
        defense={t: -0.2 for t in teams},
        home_advantage=0.2,
        rho=-0.05,
        match_counts={t: 50 for t in teams},
    )
    elo = {t: 1800 for t in teams}

    r32 = [
        _match(i + 1, teams[2 * i], teams[2 * i + 1], "2026-06-28", home_score=2, away_score=0)
        for i in range(16)
    ]
    winners = [teams[2 * i] for i in range(16)]  # each match's home team wins

    # infer_next_round_pairs would pair winners[i] with winners[i+8] (mirrored
    # top/bottom half). Deliberately provide a different real pairing instead.
    real_pairs = [(winners[i], winners[15 - i]) for i in range(8)]
    inferred_pairing = {frozenset((winners[i], winners[i + 8])) for i in range(8)}
    assert {frozenset(p) for p in real_pairs} != inferred_pairing  # sanity-check the test setup

    r16 = [
        _match(100 + i, home, away, "2026-07-04", status="TIMED", stage="LAST_16")
        for i, (home, away) in enumerate(real_pairs)
    ]
    bracket = {"matches": r32 + r16}

    calls = []
    original = bracket_module.infer_next_round_pairs

    def spy(matches):
        calls.append(sorted(m["match_id"] for m in matches))
        return original(matches)

    monkeypatch.setattr(bracket_module, "infer_next_round_pairs", spy)

    simulate_tournament(model, elo, bracket=bracket, n_iterations=3, seed=1)

    # Must never fall back to inference for the LAST_32->LAST_16 transition,
    # since the real pairing was already fully known.
    r32_ids = sorted(m["match_id"] for m in r32)
    assert r32_ids not in calls
