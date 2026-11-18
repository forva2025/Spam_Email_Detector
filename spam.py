#import relevant libraries
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset (using a spam email dataset)
# Here we'll use a common dataset available on Kaggle or UCI, stored locally as '
data = pd.read_csv('spam.csv', encoding='latin-1')

#print(data.head())  # Display the first few rows of the dataset

#data preprocessing
data = data[['v1', 'v2']]  # We are only interested in the
data.columns = ['label', 'message']  # Rename the columns for clarity
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Map 'ham' (non-spam) to 0 and 'spam' to 1

#split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42) #test size is 20% of the dataset and random state is set to 42 for reproducibility

#convert the text data to feature vectors
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)  # Fit and transform the training data
x_test_vec = vectorizer.transform(x_test)  # Only transform the test data

#train the Naive Bayes classifier
model = MultinomialNB()
model.fit(x_train_vec, y_train)

#make predictions on the test set
y_pred = model.predict(x_test_vec)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)   
conf_matrix = confusion_matrix(y_test, y_pred)

#output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

#save the model and vectorizer for future use
joblib.dump(model, 'spam_model.joblib')