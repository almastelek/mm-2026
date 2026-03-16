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
from .features import build_merged_dataset, get_feature_columns


def prepare_xy(df: pd.DataFrame):
    feats = get_feature_columns()
    use = [c for c in feats if c in df.columns]
    X = df[use].astype(float).fillna(0)
    y = df["winner"].astype(int)
    return X, y, use


def train_model(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 5,
) -> dict:
    if df is None:
        df = build_merged_dataset(years=[2023, 2024, 2025])
    X, y, feature_names = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=random_state, C=0.5)
    model.fit(X_train, y_train)

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
        "accuracy_holdout": float(acc),
        "log_loss_holdout": float(ll),
        "brier_score_holdout": float(brier),
        "cv_accuracy_mean": float(cv_res["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_res["test_accuracy"].std()),
        "cv_log_loss_mean": float(-cv_res["test_neg_log_loss"].mean()),
        "cv_log_loss_std": float(cv_res["test_neg_log_loss"].std()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": feature_names,
    }

    # Calibration: bin predicted probs and compare to actual frequency
    prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=5)
    report["calibration_bins_true"] = prob_true.tolist()
    report["calibration_bins_pred"] = prob_pred.tolist()

    # Save model and report
    import joblib
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": feature_names},
        MODELS_DIR / "game_classifier.joblib",
    )
    with open(OUTPUT_DIR / "training_report.json", "w") as f:
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
    report = train_model()
    print_report(report)
