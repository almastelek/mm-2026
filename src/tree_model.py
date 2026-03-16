"""
Train XGBoost game-level models (core features) for:
- all rounds
- high-stakes rounds (Sweet 16 and beyond; round <= 16)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

from .config import MODELS_DIR, OUTPUT_DIR
from .features import build_merged_dataset, get_core_feature_columns
from .train import prepare_xy


def train_xgb(
    df: pd.DataFrame | None = None,
    stakes: str = "all",  # all|high
    random_state: int = 42,
) -> dict:
    if df is None:
        df = build_merged_dataset(years=None)

    stakes = stakes.lower().strip()
    if stakes not in {"all", "high"}:
        raise ValueError("stakes must be 'all' or 'high'")

    if stakes == "high":
        df = df[df["round"].astype(int) <= 16]

    feature_names = get_core_feature_columns()
    X, y, used_features = prepare_xy(df, feature_names)

    years = df["year"].astype(int)
    rounds = df["round"].astype(int)
    year_max = int(years.max())
    lambda_year = 0.08
    w_year = np.exp(-lambda_year * (year_max - years))
    round_weights = {64: 1.0, 32: 1.1, 16: 1.3, 8: 1.5, 4: 1.7, 2: 2.0}
    w_round = rounds.map(round_weights).fillna(1.0)

    # Seed-gap-aware weighting (early rounds only: R64/R32):
    # de-emphasize huge dog upsets, emphasize favorites holding serve.
    # We use abs_seed_diff + which side was the favorite to build a modifier.
    if "abs_seed_diff" in df.columns:
        seed_gap = df["abs_seed_diff"].astype(float)
        winner = df["winner"].astype(int)
        seed_a = df["seed_a"].astype(int)
        seed_b = df["seed_b"].astype(int)
        rd = df["round"].astype(int)
        # Base modifier = 1.0; adjust only for big gaps.
        w_seed = np.ones_like(seed_gap, dtype=float)
        # For very large gaps (>= 10), downweight when the underdog wins, upweight when favorite wins.
        big_gap = (seed_gap >= 10) & (rd.isin([64, 32]))
        fav_is_a = seed_a < seed_b
        fav_wins = big_gap & ((fav_is_a & (winner == 1)) | ((~fav_is_a) & (winner == 0)))
        dog_wins = big_gap & (~fav_wins)
        w_seed[fav_wins] *= 1.2
        w_seed[dog_wins] *= 0.6
    else:
        w_seed = np.ones(len(df), dtype=float)

    sample_weight = (w_year * w_round * w_seed).astype(float).values

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weight, test_size=0.2, random_state=random_state, stratify=y
    )

    # Try a small set of conservative hyperparameter configs and pick best log loss.
    configs = [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5, "gamma": 0.0, "reg_lambda": 1.0},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5, "gamma": 1.0, "reg_lambda": 2.0},
        {"n_estimators": 250, "max_depth": 2, "learning_rate": 0.05, "min_child_weight": 3, "gamma": 0.5, "reg_lambda": 1.5},
    ]

    best_model = None
    best_report = None
    best_ll = np.inf

    for cfg in configs:
        model = XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            min_child_weight=cfg["min_child_weight"],
            gamma=cfg["gamma"],
            reg_lambda=cfg["reg_lambda"],
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=4,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        p_test = model.predict_proba(X_test)[:, 1]
        y_pred = (p_test >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, p_test)
        brier = brier_score_loss(y_test, p_test)

        report = {
            "stakes": stakes,
            "config": cfg,
            "accuracy_holdout": float(acc),
            "log_loss_holdout": float(ll),
            "brier_score_holdout": float(brier),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": used_features,
        }

        if ll < best_ll:
            best_ll = ll
            best_model = model
            best_report = report

    report = best_report

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    import joblib

    joblib.dump(
        {"model": best_model, "feature_names": used_features},
        MODELS_DIR / f"game_xgb_core_{stakes}.joblib",
    )
    with open(OUTPUT_DIR / f"tree_report_core_{stakes}.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_report(report: dict) -> None:
    print("=== XGBoost core model ===")
    print(f"Stakes:             {report['stakes']}")
    print(f"Holdout accuracy:   {report['accuracy_holdout']:.4f}")
    print(f"Holdout log loss:   {report['log_loss_holdout']:.4f}")
    print(f"Brier score:        {report['brier_score_holdout']:.4f}")
    print(f"n_train/n_test:     {report['n_train']}/{report['n_test']}")
    print(f"Features:           {report['features']}")


if __name__ == "__main__":
    for s in ("all", "high"):
        rpt = train_xgb(stakes=s)
        print_report(rpt)

