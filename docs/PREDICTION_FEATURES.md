## Features Required at Prediction Time

This document lists the **inputs and feature sources** needed to run game-level predictions and bracket simulation for an upcoming March Madness tournament (e.g. 2026). It reflects the current code in `src/features.py`, `src/train.py`, `src/tree_model.py`, and `src/simulate.py`.

## 1. Bracket input

- **Source:** Bracket for the target year (who plays whom in the round of 64).
- **Format:** Same structure as `data/Tournament Simulation.csv`: for the prediction year, one row per team in the round of 64, with columns at least:
  - `YEAR`
  - `BY ROUND NO` (or equivalent ordering so that consecutive pairs define games)
  - `TEAM`
  - `SEED`
- **Usage:** `src.simulate.load_bracket_first_round(year)` expects this file (or a path you pass) to list all 64 teams for that year and derive the 32 first-round matchups in bracket order.

## 2. Team–season feature sources (by YEAR + TEAM)

These must be available for the **same season** as the tournament (e.g. 2025–26 for 2026 March Madness). Join key: **normalized team name** and **year** (see `src.team_utils.team_year_key`).


| Data source                    | File (current project)         | Columns used for features |
| ----------------------------- | ------------------------------ | ------------------------- |
| **Barttorvik** (neutral)       | `Barttorvik Neutral.csv`       | `BARTHAG`, `BADJ EM`, `WAB` |
| **Resumes**                    | `Resumes.csv`                  | `ELO`, `R SCORE`, `RESUME` |
| **538** (optional)             | `538 Ratings.csv`              | `POWER RATING` |
| **KenPom (subset)**            | `KenPom Barttorvik.csv`        | `KADJ O`, `KADJ D`, `KADJ EM`, `K TEMPO`, `KADJ T`, `ELITE SOS` |
| **Teamsheet ranks (enhanced)** | `Teamsheet Ranks.csv`          | `NET`, `KPI`, `SOR`, `Q1 W`, `Q2 W`, `Q1&2 W` |


- **Required for core models:** Barttorvik + Resumes + KenPom subset (in practice, we only hard-require Barttorvik + Resumes at training time; KenPom fields will be used when present).
- **Optional:** 538 `POWER RATING`; if missing, the pipeline fills `power_rating_diff` with 0.
- **Enhanced-only:** Teamsheet ranks are historically incomplete in this repo; we handle this by training separate **core** vs **enhanced** logistic models and dynamically routing in simulation when Teamsheet features exist.

## 3. Context / lookup tables (not year-specific)

- **Seed Results** (`Seed Results.csv`): columns `SEED`, `WIN%` (or equivalent). Used to compute `seed_win_pct_a`, `seed_win_pct_b` and historical seed context. Can be the same file across years.
- **Team name normalization:** `src.team_utils.normalize_team_name` and `TEAM_ALIASES` ensure consistent joins. Add aliases if new data uses different spellings (e.g. "St. John's" vs "St John's").

## 4. Feature list used by the models

The game-level models use **difference and context** features derived from the above. Exact names are in `src.features.get_feature_columns()`; practically you’ll interact with:

- **Core feature set**: `src.features.get_core_feature_columns()` (drops Teamsheet diffs)
- **Enhanced feature set**: `src.features.get_enhanced_feature_columns()` (includes Teamsheet diffs)

**Core + enhanced features (high level):**

- **Seed/round context**
  - `seed_diff`, `abs_seed_diff`, `is_1_16`, `is_2_15`, `is_3_14`, `is_4_13`, `is_5_12`
  - `seed_win_pct_a`, `seed_win_pct_b` (from `Seed Results.csv`)
  - `round_num`
- **Team strength diffs (A − B)**
  - Barttorvik: `barthag_diff`, `badj_em_diff`
  - Resumes: `elo_diff`, `r_score_diff`
  - 538: `power_rating_diff` (0-filled if missing)
  - KenPom: `k_adj_em_diff`, `k_adj_o_diff`, `k_adj_d_diff`, `k_tempo_diff`, `k_adj_t_diff`
  - KenPom SOS: `elite_sos_diff`
- **Historical upset prior**
  - `historical_upset_prob` (empirical \(P(\text{upset} \mid \text{round}, \text{seed_diff})\) computed from all historical games)

**Enhanced-only Teamsheet diffs (A − B):**

- `net_diff`, `kpi_diff`, `sor_diff`, `q1_wins_diff`, `q12_wins_diff`

**Training-time SOS emphasis (linear models):**

- `src.train.prepare_xy()` scales a few schedule-strength style columns (when present) to give them more influence in logistic regression:
  - `elite_sos_diff`, `sor_diff`, `kpi_diff`, `net_diff`

For a **new season**, ensure the team-season files above are updated for that season’s teams and year, and that the bracket file (Tournament Simulation–style) is available for the prediction year. Then re-run the pipeline (feature build → model predict → bracket simulation) without changing code.

## 5. Models, ensembling, and saved artifacts

We currently support:

- **Logistic regression models** (core/enhanced × all/high-stakes):
  - `models/game_classifier_core_all.joblib`
  - `models/game_classifier_core_high.joblib`
  - `models/game_classifier_enhanced_all.joblib`
  - `models/game_classifier_enhanced_high.joblib`
- **XGBoost models** (core only × all/high-stakes):
  - `models/game_xgb_core_all.joblib`
  - `models/game_xgb_core_high.joblib`

During simulation, `src.simulate` can run in three strategies:

- `strategy="logistic"`: pure logistic
- `strategy="xgb"`: pure XGBoost (if available)
- `strategy="avg"`: average logistic + XGBoost when both exist (default)

- **Models:** `models/game_classifier.joblib` and/or `models/ensemble.joblib` (trained on historical data). Load and use for `predict_proba` in `src.simulate.predict_game_prob` and `run_simulation`.
- **Outputs:** `output/win_probs_<year>_<strategy>.csv` — win probabilities by round (R64, R32, S16, E8, F4, FINALS, CHAMP) per team.

## 6. Quick checklist for a new tournament year

1. Update or add CSVs for that season: Barttorvik Neutral, Resumes, (optional) 538, for the same **YEAR** as the tournament.
2. Add the bracket (round-of-64 matchups) in Tournament Simulation format for that **YEAR**.
3. (Recommended) Retrain:
   - `python -m src.train` (logistic core/enhanced × all/high)
   - `python -m src.tree_model` (XGBoost core × all/high)
4. Run simulations, choosing a strategy:
   - `python -m src.simulate --year YEAR --n_sims 10000 --strategy logistic`
   - `python -m src.simulate --year YEAR --n_sims 10000 --strategy xgb`
   - `python -m src.simulate --year YEAR --n_sims 10000 --strategy avg`
5. Compare the resulting CSVs in `output/`.

