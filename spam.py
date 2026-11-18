#import relevant libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset (using a spam email dataset)
# Here we'll use a common dataset available on Kaggle or UCI, stored locally as '
data = pd.read_csv('spam.csv', encoding='latin-1')

print(data.head())  # Display the first few rows of the dataset