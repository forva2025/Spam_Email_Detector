# Spam Email Detector

A Multinomial Naive Bayes spam classifier trained on the UCI Spambase dataset,
with a Streamlit interface for checking pasted email text.

## Run the app

```powershell
python -m pip install -r requirements.txt
python spam.py
python -m streamlit run app.py
```

The training command creates `spam_model.joblib`. The Streamlit app converts
pasted email text into the same 57 numeric features used during training.

The checker is a demonstration model, not a replacement for a mail provider's
security filters. It does not inspect URLs, attachments, sender reputation, or
malware.
