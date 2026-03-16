import pandas as pd
import joblib
from pathlib import Path

from src.simulate import load_bracket_first_round, predict_game_prob
from src.features import (
    load_barttorvik_neutral,
    load_resumes,
    load_538,
    load_kenpom_barttorvik,
    load_teamsheet_ranks,
    load_seed_results,
    load_true_upset_rates_from_games,
)
from src.config import MODELS_DIR, DATA_DIR
from src.train import scale_features


YEAR = 2026
STRATEGY = "xgb"  # "logistic" | "logistic_raw" | "xgb" | "avg"


def build_model_package(strategy: str):
    def _load_optional(p: Path):
        return joblib.load(p) if p.exists() else None

    core_all = _load_optional(MODELS_DIR / "game_classifier_core_all.joblib")
    core_high = _load_optional(MODELS_DIR / "game_classifier_core_high.joblib")
    enh_all = _load_optional(MODELS_DIR / "game_classifier_enhanced_all.joblib")
    enh_high = _load_optional(MODELS_DIR / "game_classifier_enhanced_high.joblib")

    xgb_core_all = _load_optional(MODELS_DIR / "game_xgb_core_all.joblib")
    xgb_core_high = _load_optional(MODELS_DIR / "game_xgb_core_high.joblib")

    if core_all is None:
        raise RuntimeError("Missing core logistic model. Run `python -m src.train` first.")

    def pick_model(row: pd.Series) -> dict:
        is_high = int(row.get("round", 64)) <= 16
        has_ts = pd.notna(row.get("net_diff")) and pd.notna(row.get("sor_diff"))

        if has_ts and enh_all is not None:
            return enh_high if (is_high and enh_high is not None) else enh_all
        return core_high if (is_high and core_high is not None) else core_all

    return {
        "model": core_all["model"],
        "feature_names": core_all["feature_names"],
        "_pick_model": pick_model,
        "_xgb_core_all": xgb_core_all,
        "_xgb_core_high": xgb_core_high,
        "_strategy": strategy,
    }


def load_feature_data():
    # For predictions, prefer YEAR-specific files under data/<YEAR>/ if present.
    year_dir = DATA_DIR / str(YEAR)

    bart_path = year_dir / "Barttorvik Neutral.csv"
    res_path = year_dir / "Resumes.csv"
    kp_path = year_dir / "KenPom Barttorvik.csv"
    ts_path = year_dir / "Teamsheet Ranks.csv"
    f538_path = year_dir / "538 Ratings.csv"

    barttorvik = load_barttorvik_neutral(path=bart_path if bart_path.exists() else None)
    resumes = load_resumes(path=res_path if res_path.exists() else None)
    f538 = load_538(path=f538_path if f538_path.exists() else None)
    kenpom = load_kenpom_barttorvik(path=kp_path if kp_path.exists() else None)
    teamsheet = load_teamsheet_ranks(path=ts_path if ts_path.exists() else None)
    upset_rates = load_true_upset_rates_from_games()
    seed_results = load_seed_results()
    if "seed_win_pct" not in seed_results.columns and "WIN%" in seed_results.columns:
        seed_results = seed_results.copy()
        seed_results["seed_win_pct"] = pd.to_numeric(
            seed_results["WIN%"].astype(str).str.replace("%", ""), errors="coerce"
        ).where(lambda x: x <= 1, lambda x: x / 100)
    return barttorvik, resumes, f538, kenpom, teamsheet, upset_rates, seed_results


def predict_round(
    matchups,
    round_num: int,
    model_package: dict,
    barttorvik: pd.DataFrame,
    resumes: pd.DataFrame,
    f538: pd.DataFrame,
    seed_results: pd.DataFrame,
    cache: dict,
):
    """
    matchups: list[(team, seed)] pairs, length must be even.
    Returns winners list[(team, seed)] and list of row dicts with probabilities and winners.
    """
    winners = []
    rows = []
    for i in range(0, len(matchups), 2):
        (a, seed_a), (b, seed_b) = matchups[i], matchups[i + 1]
        p_a = predict_game_prob(
            team_a=a,
            team_b=b,
            seed_a=seed_a,
            seed_b=seed_b,
            year=YEAR,
            round_num=round_num,
            model_package=model_package,
            barttorvik=barttorvik,
            resumes=resumes,
            f538=f538,
            seed_results=seed_results,
            cache=cache,
        )
        winner = (a, seed_a) if p_a >= 0.5 else (b, seed_b)
        rows.append(
            {
                "round": round_num,
                "team_a": a,
                "seed_a": seed_a,
                "team_b": b,
                "seed_b": seed_b,
                "p_team_a_wins": float(p_a),
                "p_team_b_wins": float(1.0 - p_a),
                "winner": winner[0],
                "winner_seed": winner[1],
            }
        )
        winners.append(winner)
    return winners, rows


def main():
    model_package = build_model_package(STRATEGY)
    (
        barttorvik,
        resumes,
        f538,
        kenpom,
        teamsheet,
        upset_rates,
        seed_results,
    ) = load_feature_data()

    # Attach shared refs expected by simulate._predict_game_prob_impl
    model_package = {
        **model_package,
        "_kenpom": kenpom,
        "_teamsheet": teamsheet,
        "_upset_rates": upset_rates,
    }

    # Load R64 bracket for this YEAR.
    # If you keep 2026 data in data/2026/..., read from there;
    # otherwise, add 2026 rows to data/Tournament Simulation.csv.
    bracket_path = (DATA_DIR / str(YEAR) / "Tournament Simulation.csv")
    if not bracket_path.exists():
        # Fallback to the root file if per-year file doesn't exist
        bracket_path = None

    matchups_64 = load_bracket_first_round(YEAR, path=bracket_path)
    round_64_pairs = []
    for (a, b, seed_a, seed_b) in matchups_64:
        round_64_pairs.append((a, seed_a))
        round_64_pairs.append((b, seed_b))

    cache: dict = {}
    all_rows = []

    # R64
    winners_64, rows_64 = predict_round(
        round_64_pairs, 64, model_package, barttorvik, resumes, f538, seed_results, cache
    )
    all_rows.extend(rows_64)

    # R32
    winners_32, rows_32 = predict_round(
        winners_64, 32, model_package, barttorvik, resumes, f538, seed_results, cache
    )
    all_rows.extend(rows_32)

    # S16
    winners_16, rows_16 = predict_round(
        winners_32, 16, model_package, barttorvik, resumes, f538, seed_results, cache
    )
    all_rows.extend(rows_16)

    # E8
    winners_8, rows_8 = predict_round(
        winners_16, 8, model_package, barttorvik, resumes, f538, seed_results, cache
    )
    all_rows.extend(rows_8)

    # F4
    winners_4, rows_4 = predict_round(
        winners_8, 4, model_package, barttorvik, resumes, f538, seed_results, cache
    )
    all_rows.extend(rows_4)

    # Finals (round_num = 2)
    winners_2, rows_2 = predict_round(
        winners_4, 2, model_package, barttorvik, resumes, f538, seed_results, cache
    )
    all_rows.extend(rows_2)

    df = pd.DataFrame(all_rows).sort_values(["round"], ascending=[False])

    print(f"Deterministic bracket predictions for YEAR={YEAR}, strategy={STRATEGY}")
    for rnd in [64, 32, 16, 8, 4, 2]:
        print(f"\n=== Round {rnd} ===")
        sub = df[df["round"] == rnd]
        for _, row in sub.iterrows():
            print(
                f"{row['team_a']} ({int(row['seed_a'])}) vs "
                f"{row['team_b']} ({int(row['seed_b'])}) "
                f"=> P(A wins)={row['p_team_a_wins']:.3f}, "
                f"winner={row['winner']} ({int(row['winner_seed'])})"
            )

    champ_row = df[df["round"] == 2].iloc[0]
    print("\n=== Champion prediction ===")
    print(f"{champ_row['winner']} (seed {int(champ_row['winner_seed'])})")

    # Save full game-by-game table for inspection
    out_path = Path("output") / f"deterministic_games_{YEAR}_{STRATEGY}.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved full game-by-game table to {out_path}")


if __name__ == "__main__":
    main()

