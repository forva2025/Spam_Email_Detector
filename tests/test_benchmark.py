"""Tests for benchmark configuration and the recorded experiment artifacts."""

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmark import (
    CV_FOLDS,
    MODEL_METADATA,
    RANDOM_STATE,
    build_models,
    load_dataset,
    split_dataset,
)


ROOT = Path(__file__).parents[1]
RESULTS_PATH = ROOT / "artifacts" / "model_comparison.csv"
REPORT_PATH = ROOT / "artifacts" / "benchmark_report.json"


def test_all_models_are_configured_for_fair_comparison():
    models = build_models()

    assert list(models) == list(MODEL_METADATA)
    assert len(models) == 4
    assert CV_FOLDS == 5
    assert RANDOM_STATE == 42


def test_shared_split_is_deterministic_and_stratified():
    X, y = load_dataset()
    first = split_dataset(X, y)
    second = split_dataset(X, y)
    X_train, X_test, y_train, y_test = first

    assert X_train.index.tolist() == second[0].index.tolist()
    assert X_test.index.tolist() == second[1].index.tolist()
    assert len(X_train) == 3680
    assert len(X_test) == 921
    assert y_train.mean() == pytest.approx(y.mean(), abs=0.001)
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.001)


def test_recorded_metrics_match_detailed_report():
    results = pd.read_csv(RESULTS_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert results["model_slug"].tolist() == list(MODEL_METADATA)
    assert report["split"]["random_state"] == RANDOM_STATE
    assert report["cross_validation"]["folds"] == CV_FOLDS
    assert report["cross_validation"]["scope"] == "Training split only"

    for model_report in report["models"]:
        row = results.loc[
            results["model_slug"] == model_report["model_slug"]
        ].iloc[0]
        assert row["f1"] == pytest.approx(model_report["holdout_metrics"]["f1"])
        assert row["cv_f1_mean"] == pytest.approx(
            model_report["cross_validation"]["f1"]["mean"]
        )
        assert sum(sum(values) for values in model_report["confusion_matrix"]) == 921
