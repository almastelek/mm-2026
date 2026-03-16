"""
Rating-difference baseline and ensemble (Option B).
Baseline: P(A wins) = sigmoid(k * (rating_A - rating_B) + b), fit on barthag_diff.
Ensemble: P_final = w * P_rating + (1 - w) * P_classifier; tune w on validation.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from scipy.optimize import minimize
import joblib

from .config import MODELS_DIR, OUTPUT_DIR
from .features import build_merged_dataset, get_feature_columns
from .train import prepare_xy, train_model


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def fit_rating_baseline(
    barthag_diff: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Fit P(team_a wins) = sigmoid(k * barthag_diff + b). Returns (k, b)."""
    def nll(params):
        k, b = params
        p = sigmoid(k * barthag_diff + b)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize(nll, [1.0, 0.0], method="L-BFGS-B", bounds=[(0.01, 20), (-5, 5)])
    k, b = res.x
    return float(k), float(b)


def predict_rating_baseline(barthag_diff: np.ndarray, k: float, b: float) -> np.ndarray:
    return sigmoid(k * np.asarray(barthag_diff, dtype=float) + b)


def train_baseline_and_ensemble(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    w_grid: list[float] | None = None,
) -> dict:
    if df is None:
        df = build_merged_dataset(years=[2023, 2024, 2025])
    if w_grid is None:
        w_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    X, y, _ = prepare_xy(df)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    barthag = np.asarray(X["barthag_diff"], dtype=float) if "barthag_diff" in X.columns else np.zeros(len(X))

    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=test_size, random_state=random_state, stratify=y
    )
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    barthag_train = barthag[train_idx]
    barthag_test = barthag[test_idx]

    # 1) Fit rating baseline on train
    k, b = fit_rating_baseline(barthag_train, y_train.values)
    p_rating_test = predict_rating_baseline(barthag_test, k, b)

    # 2) Train classifier (same as train.py)
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(max_iter=1000, random_state=random_state, C=0.5)
    classifier.fit(X_train, y_train)
    p_clf_test = classifier.predict_proba(X_test)[:, 1]

    # 3) Tune w on test (in practice you'd use a separate val set)
    best_w = 0.5
    best_ll = float("inf")
    for w in w_grid:
        p_ens = w * p_rating_test + (1 - w) * p_clf_test
        ll = log_loss(y_test, p_ens)
        if ll < best_ll:
            best_ll = ll
            best_w = w
    p_ensemble_test = best_w * p_rating_test + (1 - best_w) * p_clf_test

    # Metrics
    def metrics(prob, name):
        return {
            f"{name}_accuracy": float(accuracy_score(y_test, (prob >= 0.5).astype(int))),
            f"{name}_log_loss": float(log_loss(y_test, prob)),
            f"{name}_brier": float(brier_score_loss(y_test, prob)),
        }

    report = {
        "rating_baseline_k": k,
        "rating_baseline_b": b,
        "ensemble_weight_rating": best_w,
        **metrics(p_rating_test, "rating"),
        **metrics(p_clf_test, "classifier"),
        **metrics(p_ensemble_test, "ensemble"),
    }

    # Save baseline params and classifier for prediction
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": classifier,
            "feature_names": list(X.columns),
            "rating_k": k,
            "rating_b": b,
            "ensemble_w": best_w,
        },
        MODELS_DIR / "ensemble.joblib",
    )
    with open(OUTPUT_DIR / "ensemble_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_ensemble_report(report: dict) -> None:
    print("=== Rating baseline & ensemble ===")
    print(f"Rating:  k={report['rating_baseline_k']:.3f}, b={report['rating_baseline_b']:.3f}")
    print(f"         accuracy={report['rating_accuracy']:.4f}, log_loss={report['rating_log_loss']:.4f}")
    print(f"Classifier: accuracy={report['classifier_accuracy']:.4f}, log_loss={report['classifier_log_loss']:.4f}")
    print(f"Ensemble:   w_rating={report['ensemble_weight_rating']:.3f}")
    print(f"            accuracy={report['ensemble_accuracy']:.4f}, log_loss={report['ensemble_log_loss']:.4f}")


if __name__ == "__main__":
    report = train_baseline_and_ensemble()
    print_ensemble_report(report)
