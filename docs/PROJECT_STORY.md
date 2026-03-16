## Project story (chronological)

This is a lightweight, chronological log of what I've built, why I built it that way, and the tradeoffs I accepted along the way.

### Stage 0 — Goal and constraints

- **Goal**: predict March Madness game winners and generate bracket-level win probabilities by round (R64 → Champion).
- **Constraint**: lots of input CSVs with inconsistent coverage across years, and team names that don’t always match perfectly between sources.

### Stage 1 — Build the supervised training target (game outcomes)

- **Decision**: derive a clean game-level dataset from `data/Tournament Matchups.csv`.
- **Why**: everything else (features, models, backtests) depends on having reliable X \rightarrow y rows.
- **Tradeoff**: pairing and winner-derivation logic must be correct; if it’s wrong, every downstream metric lies.
- **Implementation**: `src/games.py` parses matchups into one row per game with `team_a`, `team_b`, seeds, round, scores, and `winner`.

### Stage 2 — Make joins reliable (team name normalization)

- **Decision**: standardize team names and create a stable join key `team_year_key = (normalized_team, year)`.
- **Why**: CSVs use different spellings/punctuation; naïve joins silently drop rows or create mismatches.
- **Tradeoff**: alias maintenance is manual; if a new season introduces new variants, we need to add aliases.
- **Implementation**: `src/team_utils.py` (`normalize_team_name`, `TEAM_ALIASES`, `team_year_key`).

### Stage 3 — Start simple: game-level logistic regression baseline

- **Decision**: start with a transparent baseline (logistic regression) on difference features.
- **Why**: we needed a stable “sanity anchor” before adding complexity.
- **Tradeoff**: linear models can miss non-linear interactions and conditional effects.
- **Implementation**: `src/features.py` builds difference features; `src/train.py` trains and saves models + reports.

### Stage 4 — Add bracket simulation (turn game win probs into tournament outcomes)

- **Decision**: Monte Carlo simulate the tournament bracket using per-game win probabilities.
- **Why**: the end goal is “who wins the tournament / reaches each round,” not just single-game accuracy.
- **Tradeoff**: tournament outcomes are noisy; small probability shifts can change champion rankings noticeably.
- **Implementation**: `src/simulate.py` loads the R64 bracket and simulates full brackets, outputting `output/win_probs_<year>_<strategy>.csv`.

### Stage 5 — Fix “seed realism” issues (upset behavior)

We saw models sometimes over-promote low seeds (especially extreme seed gaps).

- **Decision**: implement an R64 “gate” that blends model output with empirical seed-gap upset rates for large gaps.
- **Why**: extreme gaps (like 1–16) should be strongly anchored by historical rates; model noise shouldn’t dominate.
- **Tradeoff**: less flexible; if a true “historic upset profile” year happens, the gate dampens it.
- **Implementation**: `src.simulate._predict_game_prob_impl` applies a seed-gap blend in the Round of 64.

### Stage 6 — Handle incomplete historical data (core vs enhanced)

Teamsheet-based resume metrics (`NET`, `SOR`, etc.) are strong, but the historical coverage in this repo is incomplete.

- **Decision**: train **core** models that don’t rely on Teamsheet, and separate **enhanced** models that do.
- **Why**: filling missing ranks with 0 is dangerous (0 could mean “best in nation”).
- **Tradeoff**: added complexity: multiple model artifacts + routing logic at prediction time.
- **Implementation**:
  - `src.features.get_core_feature_columns()` drops Teamsheet diffs
  - `src.train.py` trains `core/enhanced × all/high`
  - `src.simulate.py` routes per-game based on whether Teamsheet features exist for that matchup.

### Stage 7 — High-stakes split (late rounds behave differently)

- **Decision**: train separate “high-stakes” models for Sweet 16 and beyond.
- **Why**: later rounds are different: teams are closer in quality, and the signal/variance profile changes.
- **Tradeoff**: smaller training set for high-stakes models (higher variance), more model files to manage.
- **Implementation**: `src.train.py` trains `*_high` on `round <= 16`; `src.simulate.py` routes by round.

### Stage 8 — Bring in SOS more explicitly

- **Decision**: add KenPom’s `ELITE SOS` as a schedule-strength feature (`elite_sos_diff`) and slightly emphasize SOS-like fields during training.
- **Why**: strong mid-majors can look “too good” if schedule strength isn’t represented well; SOS helps anchor that.
- **Tradeoff**: SOS can overweight conference strength indirectly; we preferred to use SOS itself rather than a “conference tier” prior.
- **Implementation**:
  - `src/features.py` joins `ELITE SOS` from `KenPom Barttorvik.csv` and creates `elite_sos_diff`
  - `src/train.prepare_xy` scales `elite_sos_diff` (and Teamsheet SOS-ish diffs when available)

### Stage 9 — Add a tree model (XGBoost) and compare strategies

- **Decision**: train a core-feature XGBoost model alongside logistic, and allow simulation to toggle:
  - pure logistic
  - pure XGBoost
  - averaged (default)
- **Why**: trees can capture interactions; averaging can reduce model-specific failure modes.
- **Tradeoff**: trees can overfit rare upset patterns unless regularized/weighted; more dependencies (OpenMP on macOS).
- **Implementation**:
  - `src/tree_model.py` trains `models/game_xgb_core_{all,high}.joblib`
  - `src/simulate.py` supports `--strategy logistic|xgb|avg` and writes distinct outputs.

### Stage 10 — Make XGBoost more conservative (reduce low-seed inflation)

- **Decision**: tighten XGBoost hyperparameters and add seed-gap-aware weighting focused on early rounds (R64/R32).
- **Why**: early-round “huge gap upset” patterns are rare; XGBoost otherwise learns to chase them.
- **Tradeoff**: could understate a legitimately upset-heavy year; but tends to improve realism overall.
- **Implementation**: `src/tree_model.py` applies a seed-gap weighting modifier only for rounds 64 and 32 and runs a small conservative hyperparameter sweep.

### Where am I now

- A working end-to-end pipeline with:
  - robust joins
  - core vs enhanced feature sets
  - all vs high-stakes routing
  - R64 upset gate
  - logistic + XGBoost (toggleable / averageable)
  - bracket simulation outputs for comparison

