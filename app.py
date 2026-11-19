"""Streamlit interface for prediction and measured model comparison."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from benchmark import MODEL_METADATA
from feature_extractor import extract_spambase_features


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "spam_model.joblib"
MODEL_DIR = ROOT / "models"
RESULTS_PATH = ROOT / "artifacts" / "model_comparison.csv"
REPORT_PATH = ROOT / "artifacts" / "benchmark_report.json"

SAFE_EXAMPLE = """Subject: Project meeting moved to Tuesday

Hi everyone,

The project meeting has been moved to Tuesday at 10 AM. Please review the
attached notes before we meet and let me know if you have any questions.

Thanks,
Jordan"""

SPAM_EXAMPLE = """Subject: WIN FREE MONEY NOW!!!

CONGRATULATIONS! You have been selected to receive FREE credit and money.
Click now to claim your order!!! Limited offer -- remove your address today!
Visit our internet business page and receive $$$ immediately!!!"""


st.set_page_config(
    page_title="Spam Email Checker",
    page_icon="📨",
    layout="wide",
)


@st.cache_resource
def load_models() -> dict[str, object]:
    """Load benchmark models, retaining the original Naive Bayes fallback."""

    models: dict[str, object] = {}
    for metadata in MODEL_METADATA.values():
        path = MODEL_DIR / str(metadata["artifact"])
        if path.exists():
            models[str(metadata["name"])] = joblib.load(path)

    if "Naive Bayes" not in models and MODEL_PATH.exists():
        models["Naive Bayes"] = joblib.load(MODEL_PATH)
    if not models:
        raise FileNotFoundError(
            "No model artifact was found. Run `python benchmark.py` first."
        )
    return models


@st.cache_resource
def load_model():
    """Return the original default model for backward compatibility."""

    return load_models()["Naive Bayes"]


@st.cache_data
def load_benchmark_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Benchmark metrics were not found. Run `python benchmark.py` first."
        )
    return pd.read_csv(RESULTS_PATH)


@st.cache_data
def load_benchmark_report() -> dict[str, object]:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            "Benchmark report was not found. Run `python benchmark.py` first."
        )
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def use_example(text: str, key: str = "email_text") -> None:
    st.session_state[key] = text


def predict_message(models: dict[str, object], text: str) -> pd.DataFrame:
    """Return comparable predictions from every supplied classifier."""

    features = extract_spambase_features(text)
    rows: list[dict[str, object]] = []
    for name, model in models.items():
        expected_features = list(model.feature_names_in_)
        if list(features.columns) != expected_features:
            raise ValueError(f"{name} and the text feature extractor do not match.")
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        spam_probability = float(probabilities[list(model.classes_).index(1)])
        rows.append(
            {
                "Model": name,
                "Prediction": "Spam" if prediction == 1 else "Not spam",
                "Spam score": spam_probability,
            }
        )
    return pd.DataFrame(rows)


def render_email_checker(models: dict[str, object]) -> None:
    st.title("📨 Spam Email Checker")
    st.write(
        "Paste an email below to get a quick indication of whether it resembles "
        "spam. Your text is processed locally by this app."
    )

    if "email_text" not in st.session_state:
        st.session_state.email_text = ""

    selected_model = st.selectbox("Prediction model", list(models), index=0)
    st.caption("Try an example")
    example_col, safe_col, clear_col = st.columns(3)
    example_col.button(
        "Suspicious email",
        use_container_width=True,
        on_click=use_example,
        args=(SPAM_EXAMPLE,),
    )
    safe_col.button(
        "Normal email",
        use_container_width=True,
        on_click=use_example,
        args=(SAFE_EXAMPLE,),
    )
    clear_col.button(
        "Clear",
        use_container_width=True,
        on_click=use_example,
        args=("",),
    )

    with st.form("spam_check_form"):
        email_text = st.text_area(
            "Email content",
            key="email_text",
            height=260,
            placeholder="Paste the subject and message here...",
        )
        submitted = st.form_submit_button(
            "Check this email",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return
    if len(email_text.strip()) < 10:
        st.warning("Please paste a little more of the email before checking it.")
        return

    try:
        comparisons = predict_message(models, email_text)
        selected = comparisons.loc[comparisons["Model"] == selected_model].iloc[0]
        prediction = selected["Prediction"]
        spam_probability = float(selected["Spam score"])

        if prediction == "Spam":
            st.error("### Likely spam")
            st.write(
                "This message contains patterns that resemble spam in the "
                "model's training data. Avoid clicking links or sharing "
                "personal information until you verify the sender."
            )
        else:
            st.success("### Likely not spam")
            st.write(
                "The model did not find strong spam-like patterns. Still "
                "check the sender and links if anything feels unusual."
            )

        st.metric(f"{selected_model} spam score", f"{spam_probability:.1%}")
        st.progress(spam_probability)
        if len(models) > 1:
            st.subheader("Side-by-side model predictions")
            st.dataframe(
                comparisons,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Spam score": st.column_config.ProgressColumn(
                        "Spam score", min_value=0.0, max_value=1.0, format="percent"
                    )
                },
            )

        with st.expander("What does this result mean?"):
            st.write(
                "The checker looks at word and character frequencies plus "
                "capital-letter patterns. It does not inspect links, "
                "attachments, sender reputation, or malware. Scores are model "
                "estimates—not calibrated real-world risk percentages."
            )
    except Exception as error:
        st.error(f"The email could not be checked: {error}")


def render_comparison_dashboard(models: dict[str, object]) -> None:
    st.title("📊 Model Comparison Dashboard")
    st.write(
        "Compare four classifiers evaluated on the same held-out examples and "
        "the same five cross-validation folds."
    )
    try:
        results = load_benchmark_results()
        report = load_benchmark_report()
    except Exception as error:
        st.error(str(error))
        return

    metric_labels = {
        "Hold-out accuracy": "accuracy",
        "Hold-out precision": "precision",
        "Hold-out recall": "recall",
        "Hold-out F1": "f1",
        "Cross-validated F1": "cv_f1_mean",
    }
    selected_metric_label = st.selectbox(
        "Evaluation metric", list(metric_labels), index=3
    )
    metric = metric_labels[selected_metric_label]
    chart_data = results.set_index("model")[[metric]].rename(
        columns={metric: selected_metric_label}
    )
    st.bar_chart(chart_data, y=selected_metric_label, horizontal=True)

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "cv_f1_mean",
        "cv_f1_std",
    ]
    st.dataframe(
        results[display_columns].rename(
            columns={
                "model": "Model",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "cv_f1_mean": "CV F1 mean",
                "cv_f1_std": "CV F1 std",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            column: st.column_config.NumberColumn(column, format="percent")
            for column in (
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "CV F1 mean",
                "CV F1 std",
            )
        },
    )

    st.subheader("Confusion matrix")
    matrix_model = st.selectbox("Model", results["model"].tolist())
    model_report = next(
        item for item in report["models"] if item["model"] == matrix_model
    )
    matrix = pd.DataFrame(
        model_report["confusion_matrix"],
        index=["Actual: Not spam", "Actual: Spam"],
        columns=["Predicted: Not spam", "Predicted: Spam"],
    )
    st.dataframe(matrix, use_container_width=True)

    st.subheader("Compare one message across every model")
    comparison_text = st.text_area(
        "Email to compare",
        key="comparison_email_text",
        height=180,
        placeholder="Paste an email here...",
    )
    if st.button("Compare predictions", type="primary"):
        if len(comparison_text.strip()) < 10:
            st.warning("Please paste a little more of the email before comparing it.")
        else:
            try:
                predictions = predict_message(models, comparison_text)
                st.dataframe(
                    predictions,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Spam score": st.column_config.ProgressColumn(
                            "Spam score",
                            min_value=0.0,
                            max_value=1.0,
                            format="percent",
                        )
                    },
                )
            except Exception as error:
                st.error(f"The email could not be compared: {error}")

    st.info(
        "Benchmark scores measure performance on Spambase's original numeric "
        "features. Pasted-email predictions use an approximate raw-text feature "
        "extractor, so a higher benchmark score does not guarantee better "
        "real-world pasted-email predictions."
    )


try:
    available_models = load_models()
except Exception as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    page = st.radio("View", ["Email checker", "Model comparison"])
    st.header("About this checker")
    st.write(
        "This demo compares Naive Bayes, Logistic Regression, Random Forest, "
        "and SVM models trained on the UCI Spambase dataset."
    )
    st.info(
        "Treat the result as a second opinion. Never rely on it alone for "
        "security-sensitive decisions."
    )

if page == "Email checker":
    render_email_checker(available_models)
else:
    render_comparison_dashboard(available_models)
