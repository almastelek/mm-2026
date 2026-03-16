# Features Required at Prediction Time

This document lists the data and features needed to run game-level predictions and bracket simulation for an upcoming March Madness tournament (e.g. 2026).

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

| Data source | File (current project) | Columns used for features |
|-------------|-------------------------|----------------------------|
| **Barttorvik** (neutral) | `Barttorvik Neutral.csv` | `BARTHAG`, `BADJ EM`, `WAB` |
| **Resumes** | `Resumes.csv` | `ELO`, `R SCORE`, `RESUME` |
| **538** (optional) | `538 Ratings.csv` | `POWER RATING` |

- **Required for core model:** Barttorvik (`BARTHAG`, `BADJ EM`) and Resumes (`ELO`, `R SCORE`). If a team is missing from either, that game may be dropped or filled with 0 (depending on pipeline settings).
- **Optional:** 538 `POWER RATING`; if missing, the pipeline fills the difference with 0.

## 3. Context / lookup tables (not year-specific)

- **Seed Results** (`Seed Results.csv`): columns `SEED`, `WIN%` (or equivalent). Used to compute `seed_win_pct_a`, `seed_win_pct_b` and historical seed context. Can be the same file across years.
- **Team name normalization:** `src.team_utils.normalize_team_name` and `TEAM_ALIASES` ensure consistent joins. Add aliases if new data uses different spellings (e.g. "St. John's" vs "St John's").

## 4. Feature list used by the model

The game-level classifier and ensemble use **difference and context** features derived from the above. Exact names are in `src.features.get_feature_columns()`:

- `seed_diff` (seed_b − seed_a)
- `seed_win_pct_a`, `seed_win_pct_b` (from Seed Results)
- `barthag_diff`, `badj_em_diff` (Barttorvik)
- `elo_diff`, `r_score_diff` (Resumes)
- `power_rating_diff` (538; 0 if missing)
- `round_num` (64, 32, 16, 8, 4, 2)

For a **new season**, ensure the team-season files above are updated for that season’s teams and year, and that the bracket file (Tournament Simulation–style) is available for the prediction year. Then re-run the pipeline (feature build → model predict → bracket simulation) without changing code.

## 5. Saved artifacts

- **Models:** `models/game_classifier.joblib` and/or `models/ensemble.joblib` (trained on historical data). Load and use for `predict_proba` in `src.simulate.predict_game_prob` and `run_simulation`.
- **Outputs:** `output/win_probs_<year>.csv` — win probabilities by round (R64, R32, S16, E8, F4, FINALS, CHAMP) per team.

## 6. Quick checklist for a new tournament year

1. Update or add CSVs for that season: Barttorvik Neutral, Resumes, (optional) 538, for the same **YEAR** as the tournament.
2. Add the bracket (round-of-64 matchups) in Tournament Simulation format for that **YEAR**.
3. Run `src.simulate.run_simulation(year=YEAR, n_sims=...)` (and optionally re-train `src.train.train_model` / `src.ensemble.train_baseline_and_ensemble` on updated historical data).
4. Use `output/win_probs_<year>.csv` for win probabilities by round.
