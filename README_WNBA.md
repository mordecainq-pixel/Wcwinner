# wnba-predictor — WNBA Match & Player Prop Predictor

A margin/total model (basketball's answer to Dixon-Coles) for predicting WNBA
game outcomes, plus a weighted-bootstrap Monte Carlo engine for player prop
betting -- built with the same discipline as `wcwinner/`: self-calculated
Elo, empirically-verified data sources, walk-forward calibration, and no
adopted hyperparameter without a backtest to back it up.

## What runs automatically vs. what needs you

| Piece | Automation level |
|---|---|
| Team results + Elo + margin model refit | **Fully automated.** `wnba-predictor refresh` re-pulls every completed game from ESPN's public scoreboard API, recomputes Elo, and refits the margin/total model. |
| Today's/upcoming schedule | **Fully automated.** `wnba-predictor today [--date YYYY-MM-DD]` pulls the live scoreboard, including in-progress and not-yet-started games. |
| Player box scores + prop projections | **Automated, opt-in.** `wnba-predictor refresh --players` re-pulls per-player box scores (one HTTP request per game, so it's slow -- ~15 min for a 3-year window) and recomputes opponent "stats allowed" factors. Kept separate from the fast team refresh so a routine refresh doesn't take 15 minutes every time. |
| Betting odds / market comparison | **Not wired up yet** for WNBA (task pending) -- props currently need a manually-supplied line (`--line`), same spirit as the injury field below. |
| Injury/squad news | **Manual by design, off by default** -- not automated for the same reason as `wcwinner`: no free API reliably covers this. Not currently exposed as a model input for WNBA; factor it in yourself when picking props for an injury-affected player. |
| First Basket Scorer | **Not implemented.** Would need play-by-play parsing (ESPN's summary endpoint has a separate `plays` field) that hasn't been verified for data quality/coverage yet -- left out rather than shipped as an unvalidated guess. |

## Setup

```bash
pip install -e .
```

Reuses the same `.env` as `wcwinner/` (`ODDS_API_KEY`, confirmed to cover
`basketball_wnba`, though it isn't wired into the CLI yet). No separate
credentials needed for the ESPN data -- it's a public, unauthenticated API.

```bash
wnba-predictor refresh              # team data + Elo + margin model (~30s)
wnba-predictor refresh --players    # + player box scores + opponent factors (~15min)
```

## Usage

```bash
wnba-predictor predict "Las Vegas Aces" "New York Liberty" --home-advantage --spread -3.5 --total-line 165.5
wnba-predictor today                                    # today's slate + model win probabilities
wnba-predictor today --date 2026-07-10                  # any date
wnba-predictor roster "Indiana Fever"                    # recently active players
wnba-predictor props "Kelsey Mitchell" --opponent "New York Liberty" --stat points --line 17.5
wnba-predictor backtest --cutoff 2024-07-01              # held-out validation
```

## Data sources — what we verified and why

**Team results + player box scores: ESPN's public scoreboard/summary API**
(`site.api.espn.com`), no auth, no key. Confirmed live: works for historical
dates back to at least 2015, and a `dates=YYYYMMDD-YYYYMMDD` range with an
explicit `limit` pulls a whole season in one request for team results. Player
box scores need one request per game (`summary?event=<id>`) -- there's no
bulk endpoint for those, confirmed by inspecting the actual response schema
rather than assuming.

**Two real data-quality bugs caught by inspecting output, not by trusting
the pipeline:** All-Star games (`comp.type.abbreviation == "ALLSTAR"`, made-up
rosters like "Team Collier vs Team Clark") and preseason exhibitions against
non-WNBA national teams (Puerto Rico, Australia, Japan, ahead of the
Olympics) were both silently polluting Elo ratings until caught by noticing
fake team names at the top of the ratings table and tracing them back to
their actual `season.type`/`competition.type` fields.

**Team strength: self-calculated Elo**, FiveThirtyEight-style margin-of-
victory multiplier + a flat playoff K-bump, no draws (unlike soccer's
version). No public API for WNBA Elo exists, so it's computed from scratch
in `features/elo.py`.

## Model

**Margin/total model** (`model/margin_model.py`): each team gets an additive
attack/defense rating fit by maximum likelihood under a Normal likelihood on
home/away scores -- not Poisson, deliberately: WNBA teams score 70-90+
points a game, a large-count near-continuous quantity, not soccer's rare
discrete "goal" events. Structurally it mirrors Dixon-Coles closely on
purpose (same time-decay weighting, same L2 regularization, same
Elo-blended empirical-Bayes shrinkage for small-sample teams) -- only the
likelihood family and the additive (vs. log-linear) attack/defense
combination changed to match the sport.

**Time decay tuned and cross-validated across multiple cutoffs, not one:**
an initial single-cutoff sweep suggested a suspiciously short 7-day
half-life; cross-validating against 2022/2023/2024 cutoffs showed 7 days
was actually one of the *worst* choices at a different cutoff. Settled on a
20-day half-life -- nearly as good on average, far more consistent across
periods. Confirms a real, sport-specific finding though: WNBA rewards a much
shorter memory than soccer's ~3-year one, since 12-15 player rosters turn
over hard year to year.

**Calibration (Platt scaling)**: a backtest calibration table found the raw
model measurably overconfident at the high end (an ~84%-confidence bin
resolved at only ~63% observed). Fit via walk-forward out-of-sample
predictions across four non-overlapping cutoffs, validated on a genuinely
separate held-out period. `win_probability()`, `prob_over()`, and
`prob_home_covers_spread()` all route through the fitted correction.

## Player props

**Why bootstrap instead of fitting a distribution per stat**
(`model/player_model.py`): props need combos (PRA, points+rebounds, ...) and
milestones (double-double, triple-double) that all depend on the *same
game's* stats moving together -- a player's big scoring nights tend to be
their big-assist nights too. Fitting independent marginals per stat and
summing them understates that correlation. Resampling whole historical game
rows (weighted toward recent games, halflife in *games* not days since a
player's role can change from one game to the next) preserves the real
within-game correlation for free, and needs no per-stat distributional
assumption -- which matters a lot for low-count stats like blocks where a
Normal or Poisson choice would otherwise be a real judgment call.

**Opponent adjustment** comes from a "stats allowed" factor computed
directly from the player box-score data itself (each team's average
stat-allowed relative to league average, shrunk toward 1.0 for teams with
few games faced) -- not repurposed from the team model's points-only
defense rating, since rebounds/assists/steals/blocks allowed don't
necessarily track points allowed.

**Validated via probability integral transform (PIT), not taken on faith:**
for each held-out player-game, simulate using only games strictly before it,
then check where the real result lands within the simulated distribution.
First pass came back looking badly miscalibrated for low-count stats
(blocks: mean PIT 0.77) -- turned out to be a tie-handling bug in the check
itself (naive `sim <= actual` over-counts zero-heavy discrete data), not a
real model flaw. Fixed with randomized tie-breaking; on real 2025-07 cutoff
data (1433 held-out player-games) every stat then calibrates to mean PIT
0.49-0.53.

## Validation

Run `wnba-predictor backtest --cutoff 2024-07-01` to reproduce. On that
cutoff (2,738 training games, 633 held-out test games):

| Metric | Model | Naive Elo-only baseline |
|---|---|---|
| Log-loss | 0.669 | 0.730 |
| Brier score | 0.238 | 0.257 |

The baseline converts the Elo rating gap into a win probability via the
standard Elo expected-score formula -- the model beats it on both metrics.

## Project layout

```
wnba/
  config.py               paths, credentials, tunable model constants
  data/
    espn_ingest.py         historical team results (one request/season)
    espn_player_ingest.py  historical player box scores (one request/game)
    espn_live.py           today's/any date's schedule, team rosters
  features/elo.py          margin-of-victory Elo
  model/
    margin_model.py        team margin/total model + calibration
    player_model.py         player prop simulation (bootstrap + opponent factors)
  validate/
    metrics.py, backtest.py       team-model log-loss/Brier vs. naive baseline
    player_backtest.py            player-projection PIT calibration check
  cli.py                   command-line interface
  pipeline.py              the refresh-and-refit pipeline shared by the CLI
tests/wnba/                pytest suite
```

## Still pending

- Series-aware (best-of-N) playoff simulation -- currently no bracket/series
  Monte Carlo, unlike `wcwinner`'s tournament simulator.
- Odds API wiring for live market comparison and Bet Builder reuse.
- First Basket Scorer (needs play-by-play data verification first).
