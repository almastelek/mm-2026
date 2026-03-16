# March Madness Game Winner Prediction

Predictors for March Madness game-level winners and bracket-level win probabilities, using the data in `./data` and the architecture described in the plan.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

1. **Parse games** from `data/Tournament Matchups.csv` into a game-level table (winner per game):

   ```bash
   python -c "from src.games import parse_games_from_matchups, load_matchups; from src.config import OUTPUT_DIR; g = parse_games_from_matchups(load_matchups()); g.to_csv(OUTPUT_DIR / 'games.csv', index=False); print(len(g), 'games')"
   ```

2. **Train the game-level classifier** (and optionally the rating baseline + ensemble):

   ```bash
   python -m src.train          # classifier only
   python -m src.ensemble       # rating baseline + ensemble
   ```

   Reports and saved models go to `output/` and `models/`.

3. **Run bracket simulation** for a year (e.g. 2025) to get win probabilities by round:

   ```bash
   python -m src.simulate --year 2025 --n_sims 2000 --strategy avg
   ```

   Or from code:

   ```python
   from src.simulate import run_simulation
   out = run_simulation(year=2025, n_sims=5000, strategy="avg")
   # out has columns: team, R64, R32, S16, E8, F4, FINALS, CHAMP
   ```

4. **Train XGBoost models** (optional, for comparison/averaging):

   ```bash
   python -m src.tree_model
   ```

## Data and features

- **Training target:** Game outcomes derived from `Tournament Matchups` (pair rows by year and round; winner = higher score).
- **Features:** Team-season stats (Barttorvik Neutral, Resumes, KenPom subset + ELITE SOS, optional 538, optional Teamsheet ranks) joined by year + team; seed and round context. See `docs/PREDICTION_FEATURES.md`.
- **Project decisions (chronological):** See `docs/PROJECT_STORY.md`.

## Project layout

- `src/games.py` — Parse matchups into game-level table with winner.
- `src/team_utils.py` — Team name normalization and team–year key for joins.
- `src/features.py` — Build game-level dataset with joined features and differences.
- `src/train.py` — Train and validate the game-level classifier; save model and report.
- `src/tree_model.py` — Train XGBoost core models (all/high-stakes); save model and report.
- `src/ensemble.py` — Rating-difference baseline and optional ensemble.
- `src/simulate.py` — Bracket simulation and win probabilities by round.
- `data/` — Input CSVs (Tournament Matchups, Barttorvik, Resumes, 538, Seed Results, Tournament Simulation, etc.).
- `output/` — Games table, training/ensemble reports, win-prob CSVs.
- `models/` — Saved classifier and ensemble (joblib).
