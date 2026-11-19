"""Unit tests for raw-email feature extraction."""

import pytest

from feature_extractor import FEATURE_NAMES, extract_spambase_features


def test_extractor_returns_all_features():
    features = extract_spambase_features("FREE money now!!!")

    assert features.shape == (1, 57)
    assert list(features.columns) == FEATURE_NAMES
    assert features.isna().sum().sum() == 0


def test_empty_text_produces_zero_features():
    features = extract_spambase_features("")

    assert (features.iloc[0] == 0).all()


def test_known_feature_values():
    features = extract_spambase_features("FREE money!!!").iloc[0]

    assert features["word_freq_free"] == pytest.approx(50.0)
    assert features["word_freq_money"] == pytest.approx(50.0)
    assert features["char_freq_!"] == pytest.approx(100 * 3 / 13)
    assert features["capital_run_length_average"] == pytest.approx(4.0)
    assert features["capital_run_length_longest"] == 4
    assert features["capital_run_length_total"] == 4

