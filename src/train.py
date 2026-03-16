"""
Train and validate game-level classifier (Option A).
Reports accuracy, log loss, and calibration.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import warnings

from .config import MODELS_DIR, OUTPUT_DIR
from .features import (
    build_merged_dataset,
    get_core_feature_columns,
    get_enhanced_feature_columns,
)


def prepare_xy(df: pd.DataFrame, feature_names: list[str]):
    use = [c for c in feature_names if c in df.columns]
    X = df[use].astype(float).fillna(0)
    # Emphasize schedule-strength style signals a bit more.
    # (Scaling doesn't change ordering, but it makes these features matter more for linear models.)
    scale = {
        "elite_sos_diff": 1.8,
        "sor_diff": 1.4,
        "kpi_diff": 1.2,
        "net_diff": 1.1,
    }
    for col, mult in scale.items():
        if col in X.columns:
            X[col] = X[col] * float(mult)
    y = df["winner"].astype(int)
    return X, y, use


def train_model(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 5,
    mode: str = "core",  # core|enhanced
    stakes: str = "all",  # all|high
) -> dict:
    if df is None:
        # Use all available years by default (Tournament Matchups goes back to ~2008)
        df = build_merged_dataset(years=None)
    mode = mode.lower().strip()
    if mode not in {"core", "enhanced"}:
        raise ValueError("mode must be 'core' or 'enhanced'")

    feature_names = get_core_feature_columns() if mode == "core" else get_enhanced_feature_columns()

    stakes = stakes.lower().strip()
    if stakes not in {"all", "high"}:
        raise ValueError("stakes must be 'all' or 'high'")
    # High-stakes games: Sweet 16 and beyond (round <= 16)
    if stakes == "high":
        df = df[df["round"].astype(int) <= 16]

    # Enhanced training requires Teamsheet coverage; otherwise the features are mostly missing.
    if mode == "enhanced":
        required = [c for c in ["net_diff", "sor_diff", "q12_wins_diff"] if c in df.columns]
        if required:
            df = df.dropna(subset=required)

    X, y, used_features = prepare_xy(df, feature_names)
    # Sample weighting: time decay (recent years count more) + late-round emphasis
    years = df["year"].astype(int)
    rounds = df["round"].astype(int)
    year_max = int(years.max())
    lambda_year = 0.08
    w_year = np.exp(-lambda_year * (year_max - years))
    round_weights = {64: 1.0, 32: 1.1, 16: 1.3, 8: 1.5, 4: 1.7, 2: 2.0}
    w_round = rounds.map(round_weights).fillna(1.0)
    sample_weight = (w_year * w_round).astype(float).values

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weight, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=random_state, C=0.5)
    model.fit(X_train, y_train, sample_weight=w_train)

    # Validation on holdout
    p_test = model.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred_test)
    ll = log_loss(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)

    # Cross-validation on full set for stability
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_res = cross_validate(
            model, X, y, cv=cv, scoring=["accuracy", "neg_log_loss"],
            return_train_score=True,
        )

    report = {
        "mode": mode,
        "stakes": stakes,
        "accuracy_holdout": float(acc),
        "log_loss_holdout": float(ll),
        "brier_score_holdout": float(brier),
        "cv_accuracy_mean": float(cv_res["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_res["test_accuracy"].std()),
        "cv_log_loss_mean": float(-cv_res["test_neg_log_loss"].mean()),
        "cv_log_loss_std": float(cv_res["test_neg_log_loss"].std()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": used_features,
    }

    # Calibration: bin predicted probs and compare to actual frequency
    prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=5)
    report["calibration_bins_true"] = prob_true.tolist()
    report["calibration_bins_pred"] = prob_pred.tolist()

    # Save model and report
    import joblib
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": used_features},
        MODELS_DIR / f"game_classifier_{mode}_{stakes}.joblib",
    )
    with open(OUTPUT_DIR / f"training_report_{mode}_{stakes}.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_report(report: dict) -> None:
    print("=== Training report ===")
    print(f"Holdout accuracy:    {report['accuracy_holdout']:.4f}")
    print(f"Holdout log loss:    {report['log_loss_holdout']:.4f}")
    print(f"Holdout Brier score: {report['brier_score_holdout']:.4f}")
    print(f"CV accuracy:        {report['cv_accuracy_mean']:.4f} ± {report['cv_accuracy_std']:.4f}")
    print(f"CV log loss:        {report['cv_log_loss_mean']:.4f} ± {report['cv_log_loss_std']:.4f}")
    print(f"Features:           {report['features']}")


if __name__ == "__main__":
    for m in ("core", "enhanced"):
        for s in ("all", "high"):
            report = train_model(mode=m, stakes=s)
            print_report(report)
