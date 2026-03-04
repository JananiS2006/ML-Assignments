# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Dataset
data = pd.read_csv("sentiment140.csv", encoding="latin-1", header=None)

# Keep only sentiment and tweet text columns
data = data[[0, 5]]
data.columns = ["Sentiment", "Tweet"]

# Convert sentiment to binary
# 0 = Negative, 4 = Positive → convert 4 to 1
data["Sentiment"] = data["Sentiment"].replace(4, 1)

# Remove neutral if present (optional safety step)
data = data[data["Sentiment"] != 2]

# OPTIONAL: Reduce dataset size if system is slow
# data = data.sample(20000, random_state=42)

# Features and Labels
X = data["Tweet"]
y = data["Sentiment"]

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Train SVM Model
# ------------------------------

print("Training SVM model...")

model = LinearSVC(max_iter=2000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# User Input Prediction
# ------------------------------

print("\nSVM Sentiment Prediction System Ready!")

while True:
    user_input = input("\nEnter a tweet (or type 'exit' to stop): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)

    if prediction[0] == 1:
        print("Prediction: Positive 😊")
    else:
        print("Prediction: Negative 😡")