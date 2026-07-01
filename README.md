# wcwinner — World Cup 2026 Match Predictor

A Dixon-Coles adjusted Poisson model for predicting World Cup 2026 matches, fit
on the full 1872-2026 international results history, blended with
self-calculated Elo ratings, and wired up to live tournament data.

## What runs automatically vs. what needs you

This is the part that matters most, so it's up front.

| Piece | Automation level |
|---|---|
| Historical results + Elo + model refit | **Fully automated.** `python scripts/refresh_data.py` (cron-able) re-downloads the Kaggle dataset, recomputes Elo from scratch, and refits Dixon-Coles. Zero copy-pasting. |
| WC 2026 fixtures, results, group tables | **Fully automated** via football-data.org. Its free tier pre-seeds the entire 104-match bracket through the Final and fills in real team names as each round resolves — no manual step needed for normal operation. |
| Bracket *connectivity* for rounds not yet reached (which QF slot a given R16 winner lands in, etc.) | **Inferred, not confirmed.** football-data.org doesn't expose the abstract bracket tree for undetermined future rounds, so `simulate/bracket.py` infers it via the standard "mirrored draw" convention and self-checks it against whatever pairings are already known. Run `python scripts/check_bracket.py` after each knockout round completes — if it reports warnings, the deep-round percentages from `wcwinner bracket` should not be trusted until you've confirmed the official bracket manually. |
| Betting odds / market comparison | **Fully automated** via The Odds API, used only for validation — never fed into the model. |
| Squad news / injuries | **Manual by design, off by default.** No free API reliably covers this. `predict_match(..., home_strength_multiplier=0.85)` (or the CLI's `--home-strength` / `--away-strength` flags, or the Streamlit "Squad news" expander) lets you manually discount a team's expected goals for a known absence. Defaults to 1.0 (no effect). |

## Setup

```bash
pip install -e .
cp .env.example .env   # fill in your keys
```

`.env` needs:
- `FOOTBALL_DATA_API_TOKEN` — free tier at football-data.org
- `ODDS_API_KEY` — free tier at the-odds-api.com
- `API_FOOTBALL_KEY` (+ `API_FOOTBALL_HOST`) — free tier at api-sports.io direct dashboard, **or** `RAPIDAPI_KEY` if you signed up via the RapidAPI-hosted mirror instead (some network policies block the direct api-sports.io host; RapidAPI serves the same data)

Kaggle auth: `kagglehub` reads `~/.kaggle/access_token` (or `~/.kaggle/kaggle.json`) automatically; no separate config in this repo.

Then run the initial fit:

```bash
python scripts/refresh_data.py
```

## Usage

```bash
wcwinner predict Argentina Brazil --knockout      # single matchup
wcwinner bracket --iterations 10000               # Monte Carlo tournament sim
wcwinner backtest --cutoff 2024-07-01             # held-out validation
wcwinner market                                   # model vs. betting market
wcwinner refresh                                  # same as scripts/refresh_data.py
```

or the visual version:

```bash
streamlit run wcwinner/app.py
```

## Data sources — what we chose and why

**Historical + Elo base: Kaggle "International Football Results" dataset.**
Despite the "-1872-to-2017" slug it's actively maintained; at time of writing
it's current through completed matches on 2026-06-29, including the
in-progress World Cup 2026. ~49,000 matches.

**Cross-check: API-Football, demoted to a minor role.** It was originally
planned as the primary live/cross-check source, but directly querying its
free tier revealed it **cannot access the 2026 season at all**
("Free plans do not have access to this season, try from 2022 to 2024") and
also blocks the `next` fixtures parameter. Since Kaggle already covers
2022-2024 thoroughly, API-Football's only remaining use is an optional spot
check (`wcwinner.data.api_football_client.coverage_check`) — it isn't a
dependency for anything live.

**Live 2026 fixtures/bracket: football-data.org.** Verified directly against
its live free tier that WC 2026 is one of its free competitions, and that it
already exposes the correctly-seeded bracket for group stage through the
Final. This ended up carrying the entire live-data job, which wasn't the
original plan (API-Football was supposed to share it) — a good example of why
this project checks real API behavior rather than trusting a plan made from
docs and reputation alone.

**Odds/validation: The Odds API.** Confirmed working against real upcoming WC
2026 fixtures across 40+ bookmakers. 500 credits/month free; used strictly
for calibration checks (`wcwinner market` / `validate/market_compare.py`),
never as a model input.

**Team strength: self-calculated Elo, not a third-party feed.** eloratings.net
has no public API (just scrapeable `.tsv` files behind an unofficial
convention), so Elo is computed from scratch in `features/elo.py` using the
published World Football Elo Ratings methodology: tournament-weighted
K-factor (20 friendly / 30 minor / 40 qualifiers / 50 continental / 60 World
Cup), a goal-difference multiplier, and a +100 home-advantage bonus. This
keeps it internally consistent with the rest of the model and lets it
auto-update on every refresh.

## Model

**Dixon-Coles adjusted Poisson** (`model/dixon_coles.py`): each team gets an
attack and defense parameter fit by maximum likelihood against the full match
history, plus a global home-advantage parameter and the Dixon & Coles (1997)
`tau` low-score correlation correction for the 0-0/1-0/0-1/1-1 cells. The
likelihood is exponentially time-decayed (older matches count for less) and
restricted to a 10-year lookback for tractability — matches older than that
would contribute negligibly to the decayed likelihood anyway.

**Elo blending for small-sample teams**: after the MLE fit, a simple linear
regression of attack/defense against Elo rating (using only well-estimated
teams, 15+ matches) gives an "Elo-implied" prior for every team. Each team's
final attack/defense is an empirical-Bayes shrinkage blend of its raw fitted
value and that prior, weighted by its match count — so Cape Verde or Bosnia,
with a handful of noisy recent results, get pulled toward what their Elo
rating would predict rather than trusted at face value on small samples,
while Brazil or France (100+ matches) are barely shrunk at all.

**Knockout resolution** (`model/knockout.py`): regulation result comes
straight from the Dixon-Coles score matrix. A regulation draw is sent to a
tilted coin flip for extra time + penalties — deliberately *not* more Poisson
sampling, since ET/shootout scoring dynamics don't match normal-time rates.
The tilt is a small Elo-based nudge off 50/50, capped with `tanh` so even a
huge rating gap can't move it far from a coin flip.

**Feature engineering** (`features/`): Elo, decay-weighted rolling
attack/defense (last 15 matches, 6-match half-life), rest-day fatigue proxy,
and cumulative head-to-head goal differential are all computed causally
per-match. These feed CLI/app context and validation diagnostics rather than
the Dixon-Coles likelihood directly — the DC fit already has its own
time-decay over the full history plus the Elo-shrinkage described above, so
folding in a second, differently-scoped decay window wouldn't have a clear,
validated calibration benefit. Group-stage-vs-knockout stage is attached from
football-data.org's live bracket state for WC 2026 rows specifically, since
Kaggle's generic history doesn't carry round-level detail for the other
~49,000 matches.

## Monte Carlo bracket simulation

`simulate/bracket.py` runs 10,000+ iterations over the real WC 2026 bracket.
Already-finished matches use their real result; everything else samples from
the Dixon-Coles distribution (with the knockout tiebreak for draws). See the
automation table above for the bracket-connectivity caveat — run
`python scripts/check_bracket.py` periodically to confirm the inferred
pairing rule still matches reality as rounds complete.

## Validation

Run `wcwinner backtest --cutoff 2024-07-01` to reproduce. On a 2024-07 cutoff
(47,453 training matches, 2,028 held-out test matches):

| Metric | Model | Naive Elo-only baseline |
|---|---|---|
| Log-loss | 0.877 | 0.900 |
| Brier score | 0.514 | 0.529 |

The baseline converts the Elo rating gap into a win probability via the
standard Elo expected-score formula and assigns the historical league-wide
draw rate as a flat P(draw) — the model beats it on both metrics, i.e. the
fitted attack/defense structure adds real signal beyond "just use the rating
gap."

**Calibration** (`validate/calibration.py`): binning predicted home-win
probability into deciles and checking against observed frequency shows good
agreement across the full range, e.g. the 70-80% predicted bin resolved at
79.2% observed, and the 90-100% bin at 94.0% observed.

**Market comparison** (`wcwinner market`): compares model probabilities
against The Odds API's de-vigged consensus for upcoming fixtures. This
regularly surfaces real disagreements worth a second look rather than
assuming the market is right — e.g. the model gave Paraguay a 21% chance
against France in one run, where the market implied only 5.7%.

## Project layout

```
wcwinner/
  config.py              paths, credentials, tunable constants
  data/                  ingestion: Kaggle, football-data.org, API-Football, Odds API, team-name alignment
  features/               Elo, rolling form, feature table assembly
  model/                 Dixon-Coles fit + knockout tiebreak
  simulate/              single-match prediction, Monte Carlo bracket sim
  validate/              metrics, backtest, calibration, market comparison
  cli.py                 command-line interface
  app.py                 Streamlit UI
  pipeline.py            the refresh-and-refit pipeline shared by the CLI and scripts/
scripts/
  refresh_data.py        cron-able: re-download data, recompute Elo, refit model
  check_bracket.py       live bracket-state + connectivity self-check
tests/                   pytest suite for the Elo/Dixon-Coles/metrics/bracket-connectivity math
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
