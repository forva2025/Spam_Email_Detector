"""Reproducible benchmark for the four Spambase classifiers used by the app."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "spam.csv"
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
LEGACY_MODEL_PATH = ROOT / "spam_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_COLUMN = "spam"


MODEL_METADATA = OrderedDict(
    [
        (
            "naive_bayes",
            {
                "name": "Naive Bayes",
                "estimator": "MultinomialNB",
                "preprocessing": "None; Spambase inputs are already non-negative.",
                "artifact": "naive_bayes.joblib",
            },
        ),
        (
            "logistic_regression",
            {
                "name": "Logistic Regression",
                "estimator": "StandardScaler + LogisticRegression",
                "preprocessing": "StandardScaler fitted inside each training fold.",
                "artifact": "logistic_regression.joblib",
            },
        ),
        (
            "random_forest",
            {
                "name": "Random Forest",
                "estimator": "RandomForestClassifier",
                "preprocessing": "None; tree-based models do not require feature scaling.",
                "artifact": "random_forest.joblib",
            },
        ),
        (
            "svm",
            {
                "name": "SVM",
                "estimator": "StandardScaler + SVC (RBF kernel)",
                "preprocessing": "StandardScaler fitted inside each training fold.",
                "artifact": "svm.joblib",
            },
        ),
    ]
)


def build_models() -> OrderedDict[str, object]:
    """Return fresh, deterministically configured estimators."""

    return OrderedDict(
        [
            ("naive_bayes", MultinomialNB()),
            (
                "logistic_regression",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=2_000,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "svm",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            SVC(
                                kernel="rbf",
                                probability=True,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the checked-in numeric Spambase dataset."""

    data = pd.read_csv(path, encoding="latin-1")
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a {TARGET_COLUMN!r} target column.")
    if data.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    X = data.drop(columns=TARGET_COLUMN)
    y = data[TARGET_COLUMN]
    if X.shape[1] != 57:
        raise ValueError(f"Expected 57 Spambase features, found {X.shape[1]}.")
    if set(y.unique()) != {0, 1}:
        raise ValueError("The spam target must contain exactly the labels 0 and 1.")
    return X, y


def split_dataset(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the single shared held-out split used by every model."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: object) -> object:
    """Convert estimator parameters into JSON-safe values."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def run_benchmark(
    data_path: Path = DATA_PATH,
    model_dir: Path = MODEL_DIR,
    artifact_dir: Path = ARTIFACT_DIR,
) -> pd.DataFrame:
    """Train, evaluate, and persist every benchmark model and its metrics."""

    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    for slug, model in build_models().items():
        metadata = MODEL_METADATA[slug]
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())

        row: dict[str, object] = {
            "model_slug": slug,
            "model": metadata["name"],
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
            "cv_accuracy_std": cv_scores["test_accuracy"].std(),
            "cv_precision_mean": cv_scores["test_precision"].mean(),
            "cv_precision_std": cv_scores["test_precision"].std(),
            "cv_recall_mean": cv_scores["test_recall"].mean(),
            "cv_recall_std": cv_scores["test_recall"].std(),
            "cv_f1_mean": cv_scores["test_f1"].mean(),
            "cv_f1_std": cv_scores["test_f1"].std(),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        }
        rows.append(row)

        artifact_path = model_dir / str(metadata["artifact"])
        joblib.dump(model, artifact_path)
        if slug == "naive_bayes":
            # Preserve the original artifact path used by existing deployments.
            joblib.dump(model, LEGACY_MODEL_PATH)

        details.append(
            {
                "model_slug": slug,
                "model": metadata["name"],
                "estimator": metadata["estimator"],
                "preprocessing": metadata["preprocessing"],
                "parameters": {
                    key: _json_value(value)
                    for key, value in model.get_params(deep=True).items()
                },
                "artifact": str(artifact_path.relative_to(ROOT)).replace("\\", "/"),
                "holdout_metrics": {
                    metric: float(row[metric])
                    for metric in ("accuracy", "precision", "recall", "f1")
                },
                "cross_validation": {
                    metric: {
                        "mean": float(row[f"cv_{metric}_mean"]),
                        "std": float(row[f"cv_{metric}_std"]),
                        "fold_scores": [
                            float(score) for score in cv_scores[f"test_{metric}"]
                        ],
                    }
                    for metric in ("accuracy", "precision", "recall", "f1")
                },
                "confusion_matrix": matrix.astype(int).tolist(),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(artifact_dir / "model_comparison.csv", index=False)

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": data_path.name,
            "sha256": _sha256(data_path),
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "class_distribution": {
                str(label): int(count)
                for label, count in y.value_counts().sort_index().items()
            },
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "scope": "Training split only",
        },
        "models": details,
    }
    (artifact_dir / "benchmark_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Naive Bayes, Logistic Regression, Random Forest, and SVM."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to a Spambase CSV containing a spam target column.",
    )
    args = parser.parse_args()
    results = run_benchmark(data_path=args.data.resolve())
    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "cv_f1_mean",
        "cv_f1_std",
    ]
    print(results[display_columns].to_string(index=False, float_format="{:.4f}".format))
    print(f"\nModels: {MODEL_DIR}")
    print(f"Metrics: {ARTIFACT_DIR / 'model_comparison.csv'}")
    print(f"Report: {ARTIFACT_DIR / 'benchmark_report.json'}")


if __name__ == "__main__":
    main()
