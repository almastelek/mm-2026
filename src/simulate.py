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
    load_kenpom_barttorvik,
    load_resumes,
    load_teamsheet_ranks,
    load_538,
    load_seed_results,
    build_game_features,
    load_true_upset_rates_from_games,
    get_core_feature_columns,
    get_enhanced_feature_columns,
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
    kenpom: pd.DataFrame,
    teamsheet: pd.DataFrame,
    upset_rates: dict[tuple[int, int], float],
    seed_results: pd.DataFrame,
) -> pd.Series:
    """
    Build one game row for prediction using the *same feature builder as training*.
    """
    game = pd.DataFrame(
        [
            {
                "year": year,
                "round": round_num,
                "team_a": team_a,
                "team_b": team_b,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "score_a": np.nan,
                "score_b": np.nan,
                "winner": np.nan,
            }
        ]
    )
    feats = build_game_features(
        game,
        barttorvik=barttorvik,
        resumes=resumes,
        f538=f538,
        kenpom=kenpom,
        teamsheet=teamsheet,
        upset_rates=upset_rates,
        seed_results=seed_results,
    )
    return feats.iloc[0]


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
    # Optional per-game model selection (core vs enhanced, high-stakes vs all)
    picker = model_package.get("_pick_model")
    if picker is not None:
        probe_row = build_feature_row(
            team_a, team_b, seed_a, seed_b, year, round_num,
            barttorvik, resumes, f538, model_package["_kenpom"], model_package["_teamsheet"],
            model_package["_upset_rates"], seed_results,
        )
        chosen = picker(probe_row)
        # Merge chosen model/feature_names into package for scoring
        model_package = {
            **model_package,
            "model": chosen["model"],
            "feature_names": chosen["feature_names"],
            "_active_stakes": chosen.get("_stakes", "all"),
        }
    row = build_feature_row(
        team_a, team_b, seed_a, seed_b, year, round_num,
        barttorvik, resumes, f538, model_package["_kenpom"], model_package["_teamsheet"],
        model_package["_upset_rates"], seed_results,
    )
    feats = model_package["feature_names"]
    X = pd.DataFrame([row])
    # ensure all model features exist
    for c in feats:
        if c not in X.columns:
            X[c] = 0
    X = X[feats].astype(float).fillna(0)

    # Logistic prediction
    p_log = float(model_package["model"].predict_proba(X)[0, 1])

    # Optional XGBoost prediction using core feature set
    p_xgb = None
    xgb_all = model_package.get("_xgb_core_all")
    xgb_high = model_package.get("_xgb_core_high")
    if xgb_all is not None:
        active_stakes = model_package.get("_active_stakes", "all")
        use_xgb = xgb_high if (active_stakes == "high" and xgb_high is not None) else xgb_all
        if use_xgb is not None:
            xgb_feats = use_xgb["feature_names"]
            X_xgb = pd.DataFrame([row])
            for c in xgb_feats:
                if c not in X_xgb.columns:
                    X_xgb[c] = 0
            X_xgb = X_xgb[xgb_feats].astype(float).fillna(0)
            p_xgb = float(use_xgb["model"].predict_proba(X_xgb)[0, 1])

    # Combine logistic and XGBoost based on chosen strategy
    strategy = model_package.get("_strategy", "avg")
    if strategy == "logistic" or p_xgb is None:
        p = p_log
    elif strategy == "xgb":
        p = p_xgb
    else:  # "avg" (default)
        p = 0.5 * p_log + 0.5 * p_xgb

    # --- R64 gate using true historical upset rates ---
    # For large seed gaps in the round of 64, blend the model probability with the
    # empirical seed-gap upset rate computed from all historical games.
    if int(round_num) == 64 and seed_a and seed_b:
        fav_seed = int(min(seed_a, seed_b))
        dog_seed = int(max(seed_a, seed_b))
        seed_gap = dog_seed - fav_seed
        if seed_gap > 0:
            upset_rate = float(model_package["_upset_rates"].get((64, seed_gap), 0.0))
            # Convert to seed-only P(team_a wins)
            p_seed_only = upset_rate if int(seed_a) == dog_seed else (1.0 - upset_rate)

            # Blend weight alpha: stronger gate for more lopsided matchups
            if seed_gap >= 13:      # 1-14, 2-15, 1-16
                alpha = 0.80
            elif seed_gap >= 11:    # 3-14, 4-15, 5-16
                alpha = 0.60
            elif seed_gap >= 9:     # 5-14, 6-15, 7-16
                alpha = 0.40
            else:
                alpha = 0.0

            if alpha > 0:
                p = alpha * p_seed_only + (1.0 - alpha) * p
    # Optional: blend with rating baseline if ensemble
    if (
        model_package.get("rating_k") is not None
        and model_package.get("rating_b") is not None
        and model_package.get("ensemble_w") is not None
    ):
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
    """
    Return P(team_a wins), with optional cache.

    Cache key includes teams + round + seeds because the R64 gate depends on seed gap.
    """
    t_lo, t_hi = (team_a, team_b) if team_a <= team_b else (team_b, team_a)
    s_lo, s_hi = (seed_a, seed_b) if seed_a <= seed_b else (seed_b, seed_a)
    key = (t_lo, t_hi, round_num, int(s_lo), int(s_hi))
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
    strategy: str = "avg",  # "logistic" | "xgb" | "avg"
) -> pd.DataFrame:
    """
    Run n_sims bracket simulations for given year; return DataFrame with columns
    team, R64, R32, S16, E8, F4, FINALS (win probabilities as fraction).
    """
    strategy = strategy.lower().strip()
    if strategy not in {"logistic", "xgb", "avg"}:
        raise ValueError("strategy must be one of 'logistic', 'xgb', or 'avg'")

    rng = np.random.default_rng(seed)
    matchups_64 = load_bracket_first_round(year)
    barttorvik = load_barttorvik_neutral()
    resumes = load_resumes()
    f538 = load_538()
    kenpom = load_kenpom_barttorvik()
    teamsheet = load_teamsheet_ranks()
    upset_rates = load_true_upset_rates_from_games()
    seed_results = load_seed_results()
    if "seed_win_pct" not in seed_results.columns:
        seed_results["seed_win_pct"] = pd.to_numeric(
            seed_results["WIN%"].astype(str).str.replace("%", ""), errors="coerce"
        ).where(lambda x: x <= 1, lambda x: x / 100)

    # Load four models: core/enhanced x all/high-stakes (S16+).
    def _load_optional(p: Path):
        return joblib.load(p) if p.exists() else None

    # Logistic models
    core_all = _load_optional(MODELS_DIR / "game_classifier_core_all.joblib")
    core_high = _load_optional(MODELS_DIR / "game_classifier_core_high.joblib")
    enh_all = _load_optional(MODELS_DIR / "game_classifier_enhanced_all.joblib")
    enh_high = _load_optional(MODELS_DIR / "game_classifier_enhanced_high.joblib")

    # XGBoost models (core only)
    xgb_core_all = _load_optional(MODELS_DIR / "game_xgb_core_all.joblib")
    xgb_core_high = _load_optional(MODELS_DIR / "game_xgb_core_high.joblib")

    # Backward compatibility fallbacks
    if core_all is None and (MODELS_DIR / "game_classifier_core.joblib").exists():
        core_all = joblib.load(MODELS_DIR / "game_classifier_core.joblib")
    if core_all is None and (MODELS_DIR / "game_classifier.joblib").exists():
        core_all = joblib.load(MODELS_DIR / "game_classifier.joblib")

    if core_all is None:
        raise FileNotFoundError("Missing core model. Run `python -m src.train` first.")

    # Attach shared data caches into model packages (not saved)
    for pkg in [core_all, core_high, enh_all, enh_high]:
        if pkg is None:
            continue
        pkg["_kenpom"] = kenpom
        pkg["_teamsheet"] = teamsheet
        pkg["_upset_rates"] = upset_rates

    def pick_model(row: pd.Series) -> dict:
        is_high = int(row.get("round", 64)) <= 16
        has_ts = pd.notna(row.get("net_diff")) and pd.notna(row.get("sor_diff"))

        if has_ts and enh_all is not None:
            return enh_high if (is_high and enh_high is not None) else enh_all
        return core_high if (is_high and core_high is not None) else core_all

    # Get all teams from bracket
    teams = set()
    for team_a, team_b, _, _ in matchups_64:
        teams.add(team_a)
        teams.add(team_b)

    counts = {t: {r: 0 for r in [64, 32, 16, 8, 4, 2, 1]} for t in teams}
    cache = {}
    for _ in range(n_sims):
        # run bracket using core model package; per-game model selection happens in _predict_game_prob_impl
        # by swapping the model_package based on feature availability.
        # We implement this by mutating a small wrapper dict per call.
        results = run_one_bracket(
            matchups_64,
            year,
            {
                "model": core_all["model"],
                "feature_names": core_all["feature_names"],
                "_kenpom": kenpom,
                "_teamsheet": teamsheet,
                "_upset_rates": upset_rates,
                "_pick_model": pick_model,
                "_xgb_core_all": xgb_core_all,
                "_xgb_core_high": xgb_core_high,
                "_strategy": strategy,
            },
            barttorvik,
            resumes,
            f538,
            seed_results,
            rng,
            cache,
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
    out.to_csv(OUTPUT_DIR / f"win_probs_{year}_{strategy}.csv", index=False)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run March Madness bracket simulations.")
    parser.add_argument("--year", type=int, default=2025, help="Tournament year to simulate.")
    parser.add_argument("--n_sims", type=int, default=2000, help="Number of Monte Carlo simulations.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="avg",
        choices=["logistic", "xgb", "avg"],
        help="Which model to use: pure logistic, pure XGBoost, or averaged.",
    )
    args = parser.parse_args()

    out = run_simulation(year=args.year, n_sims=args.n_sims, strategy=args.strategy)
    print(f"Win probabilities ({args.year}, {args.n_sims} sims, strategy={args.strategy}):")
    print(out.sort_values("CHAMP", ascending=False).head(12).to_string())
