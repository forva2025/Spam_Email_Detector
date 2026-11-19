# import relevant libraries
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the UCI Spambase dataset.
# It contains 57 precomputed numeric email features and a binary target named "spam".
data = pd.read_csv('spam.csv', encoding='latin-1')

# Separate the numeric features from the target (0 = not spam, 1 = spam).
X = data.drop(columns='spam')
y = data['spam']

# Split the dataset while preserving the spam/non-spam ratio in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train the Naive Bayes classifier directly on the precomputed features.
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set.
y_pred = model.predict(X_test)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)   
conf_matrix = confusion_matrix(y_test, y_pred)

#output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Save both the original deployment path and the dashboard's model path.
root = Path(__file__).resolve().parent
model_directory = root / "models"
model_directory.mkdir(exist_ok=True)
joblib.dump(model, root / "spam_model.joblib")
joblib.dump(model, model_directory / "naive_bayes.joblib")
