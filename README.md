# Spam Email Detector

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit 1.45.1](https://img.shields.io/badge/Streamlit-1.45.1-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn 1.7.0](https://img.shields.io/badge/scikit--learn-1.7.0-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Live app](https://img.shields.io/badge/Live_App-Open-00A67E?logo=streamlit&logoColor=white)](https://spamemaildetector-mhbu4yj9c32zrn6ohxsvbm.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine-learning project that classifies pasted email content as **likely
spam** or **likely not spam**. The classifier is trained on the UCI Spambase
dataset and is presented through a simple Streamlit web interface intended for
non-technical users.

## Live demo

Try the deployed model in your browser:

**[Open the Spam Email Detector on Streamlit Community Cloud](https://spamemaildetector-mhbu4yj9c32zrn6ohxsvbm.streamlit.app/)**

> This project is an educational demonstration. It should be used as a second
> opinion, not as a replacement for a mail provider's security filters.

## Table of contents

- [Live demo](#live-demo)
- [Features](#features)
- [Application screenshots](#application-screenshots)
- [How it works](#how-it-works)
- [Technology stack](#technology-stack)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Using the application](#using-the-application)
- [Training the model](#training-the-model)
- [Model performance](#model-performance)
- [Feature engineering](#feature-engineering)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Possible improvements](#possible-improvements)
- [License](#license)

## Features

- Paste an email's subject and body into an easy-to-use browser interface.
- Receive a clear **likely spam** or **likely not spam** result.
- View the model's spam score as a percentage and progress bar.
- Load built-in normal and suspicious example messages.
- Convert ordinary email text into the model's 57 required numeric features
  automatically.
- Validate short or empty input before attempting a prediction.
- Process pasted text locally; the application does not send it to an external
  prediction API.
- Retrain the model from the included dataset with a reproducible train/test
  split.

## Application screenshots

### Email checker

The landing screen lets a user paste an email or load one of the built-in
examples.

![Spam Email Checker landing screen](assets/streamlit-home.png)

### Spam classification

After checking a suspicious message, the interface displays a clear result and
plain-language safety guidance.

![Likely spam classification in the Streamlit app](assets/streamlit-spam-classification.png)

### Spam score

The result also includes the model's spam score and an expandable explanation
of what the score means.

![Spam score and result explanation](assets/streamlit-spam-result.png)

## How it works

```text
Pasted email text
        |
        v
Word, character, and capital-run feature extraction
        |
        v
57 numeric Spambase features in the trained column order
        |
        v
Multinomial Naive Bayes model
        |
        v
Classification and spam score displayed in Streamlit
```

The training workflow and application workflow share the same feature schema.
Before prediction, the application verifies that the generated columns match
the feature names stored in the trained model.

## Technology stack

| Technology | Version | Purpose |
| --- | ---: | --- |
| Python | 3.13 tested | Application and model-training language |
| Streamlit | 1.45.1 | Interactive web user interface |
| pandas | 2.3.0 | Dataset loading and model-ready tabular features |
| scikit-learn | 1.7.0 | Data splitting, Multinomial Naive Bayes, and evaluation |
| joblib | 1.5.1 | Saving and loading the trained model |
| Regular expressions | Python standard library | Token and uppercase-run extraction |
| Git | Any recent version | Source control |

The dependency versions are pinned in `requirements.txt` to make the local
environment reproducible.

## Dataset

The project uses the [UCI Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase).
The local `spam.csv` contains:

- **4,601 email records**
- **57 numeric input features**
- **1 binary target column** named `spam`
- **2,788 non-spam records** (`spam = 0`)
- **1,813 spam records** (`spam = 1`)

The features describe how frequently selected words and characters occur and
how uppercase-letter sequences are distributed. The dataset does not contain
the original raw email messages.

## Project structure

```text
Spam_Email_Detector/
|-- app.py                 # Streamlit user interface and prediction workflow
|-- assets/                # Screenshots used by this README
|-- feature_extractor.py   # Converts raw text into 57 Spambase-style features
|-- spam.py                # Loads data, trains, evaluates, and saves the model
|-- spam.csv               # Numeric training dataset
|-- spam_model.joblib      # Model artifact generated by spam.py
|-- requirements.txt       # Pinned Python dependencies
|-- README.md              # Project documentation
`-- LICENSE                # MIT license
```

## Getting started

### Prerequisites

- Python 3.10 or newer
- `pip`
- A terminal such as PowerShell, Command Prompt, or Bash

### 1. Clone or download the project

```bash
git clone https://github.com/forva2025/Spam_Email_Detector.git
cd Spam_Email_Detector
```

If the project is already on your computer, open a terminal in its directory
instead.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Train the model

Run this step if `spam_model.joblib` is missing or if you want to regenerate
it:

```bash
python spam.py
```

### 5. Start the web application

```bash
python -m streamlit run app.py
```

Streamlit will print a local address, normally
`http://localhost:8501`. Open that address in a browser. Stop the server with
`Ctrl+C` in the terminal.

## Using the application

1. Copy the subject and body of an email.
2. Paste them into the **Email content** field.
3. Select **Check this email**.
4. Review the classification, spam score, and safety guidance.

The **Suspicious email** and **Normal email** buttons fill the text box with
examples so the interface can be tested immediately. The **Clear** button
removes the current text.

The spam score comes from the model's `predict_proba` output. It is useful for
comparison, but it is not a calibrated real-world probability or a guarantee
that a message is safe.

## Training the model

The training script performs the following steps:

1. Loads `spam.csv` with pandas.
2. Separates the `spam` target from the 57 feature columns.
3. Creates an 80% training and 20% testing split.
4. Uses `random_state=42` for reproducibility.
5. Uses stratification to preserve the spam/non-spam ratio in both splits.
6. Trains a scikit-learn `MultinomialNB` classifier.
7. Prints accuracy and a confusion matrix for the held-out test set.
8. Saves the fitted estimator to `spam_model.joblib` with joblib.

To retrain after changing `spam.py` or the dataset:

```bash
python spam.py
```

Do not load joblib model files from unknown or untrusted sources. Deserializing
an untrusted artifact can execute malicious code.

## Model performance

Using the included 80/20 stratified split, the current training script reports:

```text
Accuracy: 77.63%
Confusion Matrix:
[[458 100]
 [106 257]]
```

The confusion matrix uses this layout:

| Actual / predicted | Not spam | Spam |
| --- | ---: | ---: |
| Not spam | 458 true negatives | 100 false positives |
| Spam | 106 false negatives | 257 true positives |

A false negative is especially important in this application because it means
a spam message was classified as not spam. Accuracy therefore should not be
the only measurement considered when improving the project.

## Feature engineering

The trained model expects numeric values rather than raw text. For the
Streamlit interface, `feature_extractor.py` converts pasted text into the same
57-column schema:

- **48 word-frequency features:** percentages for terms such as `free`,
  `money`, `credit`, `business`, `meeting`, and `project`.
- **6 character-frequency features:** percentages for `;`, `(`, `[`, `!`, `$`,
  and `#`.
- **3 uppercase-run features:** average uppercase run length, longest uppercase
  run, and total uppercase characters.

Words are matched case-insensitively. The generated DataFrame is ordered using
the exact feature list expected by the model. The UI refuses to predict if the
model's stored feature names and the extractor's columns differ.

Because the original Spambase dataset contains precomputed features rather
than its source messages, the app's raw-text conversion is an approximation of
the dataset creator's preprocessing. This can affect real-world predictions.

## Limitations

- The training data was collected in the 1990s and may not represent modern
  spam language, phishing techniques, or legitimate email patterns.
- The model does not inspect sender reputation, email headers, URLs, domains,
  attachments, images, HTML structure, or malware.
- The model recognizes only a fixed set of words and characters.
- The raw-text feature extraction approximates the preprocessing used to build
  the original numeric dataset.
- Multinomial Naive Bayes assumes a simplified relationship between features
  and may produce very confident scores.
- The measured accuracy is from one held-out split, not a production or
  cross-validation benchmark.
- Non-English messages and specialized business language may perform poorly.
- Pasted messages remain in the Streamlit process while the page is active;
  avoid pasting sensitive information into deployments you do not control.

## Troubleshooting

### `spam_model.joblib was not found`

Train the model from the project directory:

```bash
python spam.py
```

Then restart Streamlit.

### `No module named streamlit`

Install the project dependencies in the active Python environment:

```bash
python -m pip install -r requirements.txt
```

### The app opens with a model/feature mismatch error

The saved model was probably trained with a different column schema. Confirm
that `spam.csv`, `spam.py`, and `feature_extractor.py` belong to the same
version of the project, then retrain the model.

### PowerShell blocks virtual-environment activation

You can run the environment's Python directly without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Possible improvements

- Compare Multinomial Naive Bayes with logistic regression, random forests, or
  gradient-boosted models.
- Add precision, recall, F1 score, ROC-AUC, and cross-validation reporting.
- Calibrate the predicted probabilities.
- Train directly on a modern raw-email dataset using TF-IDF or transformer
  embeddings.
- Add URL, sender-domain, HTML, and attachment metadata features.
- Add automated tests for feature extraction and model compatibility.
- Package the application in Docker and add continuous integration.
- Add user feedback collection for incorrectly classified messages without
  storing private email content by default.

## License

This project's source code is available under the [MIT License](LICENSE).
The dataset has its own terms and attribution requirements; consult the UCI
dataset page before redistributing it.
