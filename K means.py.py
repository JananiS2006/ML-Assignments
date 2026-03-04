# -----------------------------
# K-MEANS SENTIMENT ANALYSIS
# ----------------------------

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

# -----------------------------
# Load Dataset
# -----------------------------

print("Loading dataset...")

data = pd.read_csv("sentiment140.csv", encoding="latin-1", header=None)

# Keep only Sentiment and Tweet columns
data = data[[0, 5]].copy()
data.columns = ["Sentiment", "Tweet"]

# Convert 4 → 1
data.loc[:, "Sentiment"] = data["Sentiment"].replace(4, 1)

# Take smaller sample for faster clustering
data = data.sample(5000, random_state=42).reset_index(drop=True)

# -----------------------------
# Text Vectorization (TF-IDF)
# -----------------------------

print("Vectorizing text...")

vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X = vectorizer.fit_transform(data["Tweet"])

# -----------------------------
# Apply K-Means
# -----------------------------

print("Training K-Means model...")

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

clusters = kmeans.labels_
data["Cluster"] = clusters

# -----------------------------
# Evaluate Clustering
# -----------------------------

score = silhouette_score(X, clusters)
print("\nSilhouette Score:", score)

# -----------------------------
# Map Clusters to Sentiment
# -----------------------------

cluster_sentiment = data.groupby("Cluster")["Sentiment"].mean()

print("\nCluster Sentiment Means:")
print(cluster_sentiment)

positive_cluster = cluster_sentiment.idxmax()
negative_cluster = cluster_sentiment.idxmin()

print("\nCluster", positive_cluster, "→ Positive")
print("Cluster", negative_cluster, "→ Negative")

# -----------------------------
# Prediction Loop
# -----------------------------

print("\nK-Means Sentiment Prediction Ready!")

while True:
    user_input = input("\nEnter a tweet (or type 'exit' to stop): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    # Convert input to vector
    user_vector = vectorizer.transform([user_input])

    # Find nearest cluster center
    distances = euclidean_distances(user_vector, kmeans.cluster_centers_)
    cluster = distances.argmin()

    # Predict sentiment
    if cluster == positive_cluster:
        print("Prediction: Positive 😊")
    else:
        print("Prediction: Negative 😡")