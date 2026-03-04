# Import Libraries
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ------------------------------
# Load Dataset
# ------------------------------

data = pd.read_csv("sentiment140.csv", encoding="latin-1", header=None)

# Keep sentiment + tweet
data = data[[0, 5]]
data.columns = ["Sentiment", "Tweet"]

# Convert 4 → 1
data.loc[:, "Sentiment"] = data["Sentiment"].replace(4, 1)

# 🔥 IMPORTANT: Use small sample (Hierarchical is slow)
data = data.sample(1000, random_state=42)

# ------------------------------
# Convert Text to TF-IDF
# ------------------------------

vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
X = vectorizer.fit_transform(data["Tweet"]).toarray()

# ------------------------------
# Apply Hierarchical Clustering
# ------------------------------

print("Training Hierarchical Clustering...")

model = AgglomerativeClustering(n_clusters=2)
clusters = model.fit_predict(X)

data["Cluster"] = clusters

# ------------------------------
# Evaluate Clustering
# ------------------------------

score = silhouette_score(X, clusters)
print("\nSilhouette Score:", score)

# ------------------------------
# Map Clusters to Sentiment
# ------------------------------

cluster_sentiment = data.groupby("Cluster")["Sentiment"].mean()

print("\nCluster Sentiment Means:")
print(cluster_sentiment)

positive_cluster = cluster_sentiment.idxmax()
negative_cluster = cluster_sentiment.idxmin()

print("\nCluster", positive_cluster, "→ Positive")
print("Cluster", negative_cluster, "→ Negative")

# ------------------------------
# User Input Prediction
# ------------------------------

print("\nHierarchical Sentiment Prediction Ready!")

while True:
    user_input = input("\nEnter a tweet (or type 'exit' to stop): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    user_vector = vectorizer.transform([user_input]).toarray()
    
    # Find nearest cluster by distance
    from sklearn.metrics.pairwise import euclidean_distances
    
    distances = euclidean_distances(user_vector, X)
    nearest_index = distances.argmin()
    cluster = clusters[nearest_index]

    if cluster == positive_cluster:
        print("Prediction: Positive 😊")
    else:
        print("Prediction: Negative 😡")