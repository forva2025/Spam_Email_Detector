"""Streamlit interface for the trained Spambase classifier."""

from pathlib import Path

import joblib
import streamlit as st

from feature_extractor import extract_spambase_features


MODEL_PATH = Path(__file__).with_name("spam_model.joblib")

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
    layout="centered",
)


@st.cache_resource
def load_model():
    """Load the local model once per Streamlit process."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "spam_model.joblib was not found. Run `python spam.py` first."
        )
    return joblib.load(MODEL_PATH)


def use_example(text: str) -> None:
    st.session_state.email_text = text


st.title("📨 Spam Email Checker")
st.write(
    "Paste an email below to get a quick indication of whether it resembles "
    "spam. Your text is processed locally by this app."
)

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

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

if submitted:
    if len(email_text.strip()) < 10:
        st.warning("Please paste a little more of the email before checking it.")
    else:
        try:
            model = load_model()
            features = extract_spambase_features(email_text)

            expected_features = list(model.feature_names_in_)
            if list(features.columns) != expected_features:
                raise ValueError("The model and text feature extractor do not match.")

            prediction = int(model.predict(features)[0])
            probabilities = model.predict_proba(features)[0]
            spam_probability = float(
                probabilities[list(model.classes_).index(1)]
            )

            if prediction == 1:
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

            st.metric("Model's spam score", f"{spam_probability:.1%}")
            st.progress(spam_probability)

            with st.expander("What does this result mean?"):
                st.write(
                    "The checker looks at word and character frequencies plus "
                    "capital-letter patterns. It does not inspect links, "
                    "attachments, sender reputation, or malware. The score is "
                    "a model estimate—not a guarantee or a calibrated risk "
                    "percentage."
                )
        except Exception as error:
            st.error(f"The email could not be checked: {error}")

with st.sidebar:
    st.header("About this checker")
    st.write(
        "This demo uses a Multinomial Naive Bayes model trained on the UCI "
        "Spambase dataset. Its test accuracy in this project is 77.63%."
    )
    st.info(
        "Treat the result as a second opinion. Never rely on it alone for "
        "security-sensitive decisions."
    )

