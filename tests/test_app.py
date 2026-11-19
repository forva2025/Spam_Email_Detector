"""Headless integration tests for the Streamlit interface."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "app.py"
APP_TEST_TIMEOUT = 30


def start_app():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=APP_TEST_TIMEOUT)
    assert not app.exception
    return app


def test_app_starts_without_errors():
    app = start_app()

    assert app.title[0].value == "📨 Spam Email Checker"
    assert len(app.text_area) == 1
    assert len(app.button) == 4
    assert app.selectbox[0].value == "Naive Bayes"


def test_short_input_displays_warning():
    app = start_app()

    app.text_area[0].set_value("short")
    app.button[3].click().run(timeout=APP_TEST_TIMEOUT)

    assert not app.exception
    assert "Please paste a little more" in app.warning[0].value


def test_suspicious_example_displays_spam_result():
    app = start_app()

    app.button[0].click().run(timeout=APP_TEST_TIMEOUT)
    app.button[3].click().run(timeout=APP_TEST_TIMEOUT)

    assert not app.exception
    assert app.error[0].value == "### Likely spam"
    assert app.metric[0].label == "Naive Bayes spam score"
    assert len(app.dataframe) == 1


def test_comparison_dashboard_loads_measured_artifacts():
    app = start_app()

    app.radio[0].set_value("Model comparison").run(timeout=APP_TEST_TIMEOUT)

    assert not app.exception
    assert app.title[0].value == "📊 Model Comparison Dashboard"
    assert len(app.selectbox) == 2
    assert app.selectbox[0].value == "Hold-out F1"
    assert len(app.dataframe) == 2


def test_dashboard_compares_all_four_predictions():
    app = start_app()
    app.radio[0].set_value("Model comparison").run(timeout=APP_TEST_TIMEOUT)

    app.text_area[0].set_value("WIN FREE MONEY NOW!!! CLAIM YOUR CREDIT $$$")
    app.button[0].click().run(timeout=APP_TEST_TIMEOUT)

    assert not app.exception
    assert len(app.dataframe) == 3
