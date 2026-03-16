"""
Bracket simulation: for a given year's bracket, use game-level P(team_a wins) to run
Monte Carlo simulations and output win probabilities by round (R64, R32, S16, E8, F4, Champ).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from .config import DATA_DIR, MODELS_DIR, OUTPUT_DIR
from .features import (
    load_barttorvik_neutral,
    load_resumes,
    load_538,
    load_seed_results,
    get_feature_columns,
    build_game_features,
)
from .team_utils import team_year_key


ROUND_NAMES = {64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "F4", 2: "FINALS"}


def load_bracket_first_round(year: int, path: Path | None = None) -> list[tuple[str, str, int, int]]:
    """
    Load round-of-64 matchups from Tournament Simulation for a year.
    Returns list of (team_a, team_b, seed_a, seed_b) in bracket order (winner of 0 plays winner of 1, etc.).
    """
    path = path or (DATA_DIR / "Tournament Simulation.csv")
    df = pd.read_csv(path)
    df = df[df["YEAR"] == year]
    df = df[df["CURRENT ROUND"] == 64].sort_values("BY ROUND NO", ascending=False).reset_index(drop=True)
    n = len(df)
    if n % 2 != 0:
        raise ValueError(f"Odd number of teams for year {year}: {n}")
    out = []
    for i in range(0, n, 2):
        a = df.iloc[i]
        b = df.iloc[i + 1]
        out.append((a["TEAM"], b["TEAM"], int(a["SEED"]), int(b["SEED"])))
    return out


def build_feature_row(
    team_a: str,
    team_b: str,
    seed_a: int,
    seed_b: int,
    year: int,
    round_num: int,
    barttorvik: pd.DataFrame,
    resumes: pd.DataFrame,
    f538: pd.DataFrame,
    seed_results: pd.DataFrame,
) -> pd.Series | None:
    """Build one game row for prediction: team keys and joined features, then diffs."""
    key_a = team_year_key(team_a, year)
    key_b = team_year_key(team_b, year)

    bt = barttorvik.set_index("team_year_key")
    res = resumes.set_index("team_year_key")
    f5 = f538.set_index("team_year_key")

    row = {"year": year, "round": round_num, "team_a": team_a, "team_b": team_b, "seed_a": seed_a, "seed_b": seed_b}
    for name, key, df, cols in [
        ("a", key_a, bt, ["BARTHAG", "BADJ EM", "WAB"]),
        ("b", key_b, bt, ["BARTHAG", "BADJ EM", "WAB"]),
    ]:
        if key in df.index:
            for c in cols:
                row[c + "_" + name] = df.loc[key, c]
        else:
            for c in cols:
                row[c + "_" + name] = np.nan
    for name, key, df, cols in [
        ("a", key_a, res, ["ELO", "R SCORE", "RESUME"]),
        ("b", key_b, res, ["ELO", "R SCORE", "RESUME"]),
    ]:
        if key in df.index:
            for c in cols:
                row[c.lower().replace(" ", "_") + "_" + name] = df.loc[key, c]
        else:
            for c in cols:
                row[c.lower().replace(" ", "_") + "_" + name] = np.nan
    for name, key, df in [("a", key_a, f5), ("b", key_b, f5)]:
        if key in df.index:
            row["power_rating_" + name] = df.loc[key, "POWER RATING"]
        else:
            row["power_rating_" + name] = np.nan

    seed_lookup = seed_results.set_index("SEED")["seed_win_pct"].to_dict()
    row["seed_win_pct_a"] = seed_lookup.get(seed_a, np.nan)
    row["seed_win_pct_b"] = seed_lookup.get(seed_b, np.nan)
    row["seed_diff"] = seed_b - seed_a
    row["barthag_diff"] = row.get("BARTHAG_a", np.nan) - row.get("BARTHAG_b", np.nan)
    row["badj_em_diff"] = row.get("BADJ EM_a", np.nan) - row.get("BADJ EM_b", np.nan)
    row["elo_diff"] = row.get("elo_a", np.nan) - row.get("elo_b", np.nan)
    row["r_score_diff"] = row.get("r_score_a", np.nan) - row.get("r_score_b", np.nan)
    row["power_rating_diff"] = row.get("power_rating_a", np.nan) - row.get("power_rating_b", np.nan)
    row["round_num"] = round_num
    return pd.Series(row)


def _predict_game_prob_impl(
    team_a: str,
    team_b: str,
    seed_a: int,
    seed_b: int,
    year: int,
    round_num: int,
    model_package: dict,
    barttorvik: pd.DataFrame,
    resumes: pd.DataFrame,
    f538: pd.DataFrame,
    seed_results: pd.DataFrame,
) -> float:
    """Return P(team_a wins)."""
    row = build_feature_row(
        team_a, team_b, seed_a, seed_b, year, round_num,
        barttorvik, resumes, f538, seed_results,
    )
    if row is None:
        return 0.5
    feats = get_feature_columns()
    X = pd.DataFrame([row])
    use = [c for c in feats if c in X.columns]
    X = X[use].astype(float).fillna(0)
    if use != model_package["feature_names"]:
        # fill missing with 0
        for c in model_package["feature_names"]:
            if c not in X.columns:
                X[c] = 0
        X = X[model_package["feature_names"]]
    p = model_package["model"].predict_proba(X)[0, 1]
    # Optional: blend with rating baseline if ensemble
    if "rating_k" in model_package and "ensemble_w" in model_package:
        from .ensemble import predict_rating_baseline
        p_rating = predict_rating_baseline(
            np.array([row["barthag_diff"]]),
            model_package["rating_k"],
            model_package["rating_b"],
        )[0]
        w = model_package["ensemble_w"]
        p = w * p_rating + (1 - w) * p
    return float(p)


def predict_game_prob(
    team_a: str,
    team_b: str,
    seed_a: int,
    seed_b: int,
    year: int,
    round_num: int,
    model_package: dict,
    barttorvik: pd.DataFrame,
    resumes: pd.DataFrame,
    f538: pd.DataFrame,
    seed_results: pd.DataFrame,
    cache: dict | None = None,
) -> float:
    """Return P(team_a wins), with optional cache keyed by (min(team_a,team_b), max(team_a,team_b), round_num)."""
    key = (min(team_a, team_b), max(team_a, team_b), round_num)
    if cache is not None and key in cache:
        p = cache[key]
        return p if team_a == key[0] else 1.0 - p
    p = _predict_game_prob_impl(
        team_a, team_b, seed_a, seed_b, year, round_num,
        model_package, barttorvik, resumes, f538, seed_results,
    )
    if cache is not None:
        cache[key] = p
    return float(p)


def run_one_bracket(
    matchups_64: list[tuple[str, str, int, int]],
    year: int,
    model_package: dict,
    barttorvik: pd.DataFrame,
    resumes: pd.DataFrame,
    f538: pd.DataFrame,
    seed_results: pd.DataFrame,
    rng: np.random.Generator,
    cache: dict | None = None,
) -> list[tuple[str, int]]:
    """
    Run one full bracket: resolve each game by sampling from P(team_a wins).
    Returns list of (team_name, round_reached) for each team. round_reached = 64, 32, 16, 8, 4, 2, 1.
    """
    # Round of 64 winners as (team, original_seed)
    round_64_winners: list[tuple[str, int]] = []
    for team_a, team_b, seed_a, seed_b in matchups_64:
        p = predict_game_prob(
            team_a, team_b, seed_a, seed_b, year, 64,
            model_package, barttorvik, resumes, f538, seed_results, cache,
        )
        win_a = rng.random() < p
        if win_a:
            round_64_winners.append((team_a, seed_a))
        else:
            round_64_winners.append((team_b, seed_b))

    # Round 32: 16 games
    round_32_winners: list[tuple[str, int]] = []
    for i in range(0, 32, 2):
        (a, seed_a), (b, seed_b) = round_64_winners[i], round_64_winners[i + 1]
        p = predict_game_prob(a, b, seed_a, seed_b, year, 32, model_package, barttorvik, resumes, f538, seed_results, cache)
        win_a = rng.random() < p
        round_32_winners.append((a, seed_a) if win_a else (b, seed_b))

    # Sweet 16
    round_16_winners: list[tuple[str, int]] = []
    for i in range(0, 16, 2):
        (a, seed_a), (b, seed_b) = round_32_winners[i], round_32_winners[i + 1]
        p = predict_game_prob(a, b, seed_a, seed_b, year, 16, model_package, barttorvik, resumes, f538, seed_results, cache)
        win_a = rng.random() < p
        round_16_winners.append((a, seed_a) if win_a else (b, seed_b))

    # Elite 8
    round_8_winners: list[tuple[str, int]] = []
    for i in range(0, 8, 2):
        (a, seed_a), (b, seed_b) = round_16_winners[i], round_16_winners[i + 1]
        p = predict_game_prob(a, b, seed_a, seed_b, year, 8, model_package, barttorvik, resumes, f538, seed_results, cache)
        win_a = rng.random() < p
        round_8_winners.append((a, seed_a) if win_a else (b, seed_b))

    # Final Four
    round_4_winners: list[tuple[str, int]] = []
    for i in range(0, 4, 2):
        (a, seed_a), (b, seed_b) = round_8_winners[i], round_8_winners[i + 1]
        p = predict_game_prob(a, b, seed_a, seed_b, year, 4, model_package, barttorvik, resumes, f538, seed_results, cache)
        win_a = rng.random() < p
        round_4_winners.append((a, seed_a) if win_a else (b, seed_b))

    # Final
    (a, seed_a), (b, seed_b) = round_4_winners[0], round_4_winners[1]
    p = predict_game_prob(a, b, seed_a, seed_b, year, 2, model_package, barttorvik, resumes, f538, seed_results, cache)
    champ_team, champ_seed = ((a, seed_a) if rng.random() < p else (b, seed_b))

    # round_reached: 1=champ, 2=runner-up, 4=F4, 8=E8, 16=S16, 32=R32, 64=R64
    r4 = {t for t, _ in round_4_winners}
    r8 = {t for t, _ in round_8_winners}
    r16 = {t for t, _ in round_16_winners}
    r32 = {t for t, _ in round_32_winners}
    results: list[tuple[str, int]] = []
    # teams that advanced at least once
    for t, _ in round_64_winners:
        if t == champ_team:
            results.append((t, 1))
        elif t in r4:
            results.append((t, 2))
        elif t in r8:
            results.append((t, 4))
        elif t in r16:
            results.append((t, 8))
        elif t in r32:
            results.append((t, 16))
        else:
            results.append((t, 32))
    # R64 losers: for each game, the team that didn't win
    for i, (team_a, team_b, _, _) in enumerate(matchups_64):
        winner, _ = round_64_winners[i]
        loser = team_b if winner == team_a else team_a
        results.append((loser, 64))
    return results


def run_simulation(
    year: int,
    n_sims: int = 10000,
    seed: int = 42,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """
    Run n_sims bracket simulations for given year; return DataFrame with columns
    team, R64, R32, S16, E8, F4, FINALS (win probabilities as fraction).
    """
    rng = np.random.default_rng(seed)
    matchups_64 = load_bracket_first_round(year)
    barttorvik = load_barttorvik_neutral()
    resumes = load_resumes()
    f538 = load_538()
    seed_results = load_seed_results()
    if "seed_win_pct" not in seed_results.columns:
        seed_results["seed_win_pct"] = pd.to_numeric(
            seed_results["WIN%"].astype(str).str.replace("%", ""), errors="coerce"
        ).where(lambda x: x <= 1, lambda x: x / 100)

    model_path = model_path or (MODELS_DIR / "ensemble.joblib")
    if not model_path.exists():
        model_path = MODELS_DIR / "game_classifier.joblib"
    model_package = joblib.load(model_path)
    if not isinstance(model_package, dict):
        model_package = {"model": model_package, "feature_names": get_feature_columns()}

    # Get all teams from bracket
    teams = set()
    for team_a, team_b, _, _ in matchups_64:
        teams.add(team_a)
        teams.add(team_b)

    counts = {t: {r: 0 for r in [64, 32, 16, 8, 4, 2, 1]} for t in teams}
    cache = {}
    for _ in range(n_sims):
        results = run_one_bracket(
            matchups_64, year, model_package, barttorvik, resumes, f538, seed_results, rng, cache
        )
        for team, round_reached in results:
            # "Reached at least round r": increment all r such that round_reached <= r
            for r in [64, 32, 16, 8, 4, 2, 1]:
                if round_reached <= r:
                    counts[team][r] += 1

    rows = []
    for team in sorted(teams):
        row = {"team": team}
        for r, name in [(64, "R64"), (32, "R32"), (16, "S16"), (8, "E8"), (4, "F4"), (2, "FINALS")]:
            row[name] = counts[team][r] / n_sims
        row["CHAMP"] = counts[team][1] / n_sims
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / f"win_probs_{year}.csv", index=False)
    return out


if __name__ == "__main__":
    out = run_simulation(year=2025, n_sims=2000)
    print("Win probabilities (2025, 2000 sims):")
    print(out.sort_values("CHAMP", ascending=False).head(12).to_string())
