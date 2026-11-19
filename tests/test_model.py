"""Compatibility and prediction tests for the saved model."""

from pathlib import Path

import joblib
import pytest

from feature_extractor import FEATURE_NAMES, extract_spambase_features


MODEL_PATH = Path(__file__).parents[1] / "spam_model.joblib"


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), "Run `python spam.py` to create the model."
    return joblib.load(MODEL_PATH)


def test_model_matches_feature_extractor(model):
    assert list(model.feature_names_in_) == FEATURE_NAMES
    assert set(model.classes_) == {0, 1}


def test_model_returns_valid_prediction(model):
    features = extract_spambase_features("WIN FREE MONEY NOW!!!")

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    assert prediction in (0, 1)
    assert len(probabilities) == 2
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)


@pytest.mark.parametrize(
    ("text", "expected_prediction"),
    [
        ("WIN FREE MONEY NOW!!! CLAIM YOUR CREDIT $$$", 1),
        ("The project meeting is Tuesday. Please review the report.", 0),
    ],
)
def test_example_predictions_remain_stable(model, text, expected_prediction):
    features = extract_spambase_features(text)

    assert int(model.predict(features)[0]) == expected_prediction

