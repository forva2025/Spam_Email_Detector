"""Convert raw email text into the 57 features used by UCI Spambase."""

from __future__ import annotations

import re

import pandas as pd


WORD_FEATURES = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
    "meeting", "original", "project", "re", "edu", "table", "conference",
]

CHARACTER_FEATURES = [";", "(", "[", "!", "$", "#"]

FEATURE_NAMES = (
    [f"word_freq_{word}" for word in WORD_FEATURES]
    + [f"char_freq_{character}" for character in CHARACTER_FEATURES]
    + [
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ]
)


def extract_spambase_features(text: str) -> pd.DataFrame:
    """Return one model-ready row derived from an email's raw text.

    Frequencies are percentages, matching the representation used by the
    Spambase dataset. Capital-run features describe consecutive uppercase
    character sequences.
    """

    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    word_count = len(words)
    character_count = len(text)

    features: dict[str, float] = {}
    for word in WORD_FEATURES:
        occurrences = words.count(word)
        features[f"word_freq_{word}"] = (
            100.0 * occurrences / word_count if word_count else 0.0
        )

    for character in CHARACTER_FEATURES:
        features[f"char_freq_{character}"] = (
            100.0 * text.count(character) / character_count
            if character_count
            else 0.0
        )

    capital_runs = [len(run) for run in re.findall(r"[A-Z]+", text)]
    capital_total = sum(capital_runs)
    features["capital_run_length_average"] = (
        capital_total / len(capital_runs) if capital_runs else 0.0
    )
    features["capital_run_length_longest"] = max(capital_runs, default=0)
    features["capital_run_length_total"] = capital_total

    return pd.DataFrame([features], columns=FEATURE_NAMES)

