"""
Train shallow neural network (MLP) game-level models (core features) for:
- all rounds
- high-stakes rounds (Sweet 16 and beyond; round <= 16)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.neural_network import MLPClassifier

from .config import MODELS_DIR, OUTPUT_DIR
from .features import build_merged_dataset, get_core_feature_columns
from .train import prepare_xy


def train_mlp(
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
    sample_weight = (w_year * w_round).astype(float).values

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weight, test_size=0.2, random_state=random_state, stratify=y
    )

    # Modest MLP to reduce overfitting; early stopping on validation split.
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-3,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=random_state,
    )

    model.fit(X_train, y_train, sample_weight=w_train)

    p_test = model.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)

    report = {
        "stakes": stakes,
        "accuracy_holdout": float(acc),
        "log_loss_holdout": float(ll),
        "brier_score_holdout": float(brier),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": used_features,
        "hidden_layer_sizes": (64, 32),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    import joblib

    joblib.dump(
        {"model": model, "feature_names": used_features},
        MODELS_DIR / f"game_mlp_core_{stakes}.joblib",
    )
    with open(OUTPUT_DIR / f"nn_report_core_{stakes}.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_report(report: dict) -> None:
    print("=== MLP core model ===")
    print(f"Stakes:             {report['stakes']}")
    print(f"Holdout accuracy:   {report['accuracy_holdout']:.4f}")
    print(f"Holdout log loss:   {report['log_loss_holdout']:.4f}")
    print(f"Brier score:        {report['brier_score_holdout']:.4f}")
    print(f"n_train/n_test:     {report['n_train']}/{report['n_test']}")
    print(f"Features:           {report['features']}")


if __name__ == "__main__":
    for s in ("all", "high"):
        rpt = train_mlp(stakes=s)
        print_report(rpt)

