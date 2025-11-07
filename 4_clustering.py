import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

anime_data = pd.read_csv("data/4_anime.csv")

print(anime_data.describe())
print(anime_data.columns)

# Prepare data for clustering
numerical_cols = ['Score', 'Episodes', 'Duration_min',]
anime_data_copy = anime_data.copy()
anime_data_copy['Episodes'] = pd.to_numeric(anime_data_copy['Episodes'], errors='coerce')
anime_data_copy['Duration_min'] = anime_data_copy['Duration'].str.extract(r'(\d+)').astype(float)

new_data = anime_data_copy[numerical_cols].dropna()
print(new_data.head())

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(new_data)

new_data_with_clusters = new_data.copy()
new_data_with_clusters['kmeans_cluster'] = kmeans.labels_
print("Number of data points:", len(new_data_with_clusters))

# Elbow method to find optimal k
wss = []
for k in range(1, 16):
      kmeans = KMeans(n_clusters=k, random_state=42)
      kmeans.fit(new_data)
      wss.append(kmeans.inertia_)
      
plt.plot(range(1, 16), wss, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Total within-clusters sum of square')
plt.show()

scaler = StandardScaler()
X_std = scaler.fit_transform(new_data)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
      
# Visualize the clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

silhouette_avg = silhouette_score(new_data, new_data_with_clusters['kmeans_cluster'])
print("Mean Silhouette width for K-Means Clustering:", silhouette_avg)

agglomerative = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = agglomerative.fit_predict(new_data)

new_data_with_clusters['hierarchical_cluster'] = hierarchical_labels

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels,
cmap='viridis', alpha=0.6, s=50)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Hierarchical Clustering')
plt.show()

silhouette_avg_hierarchical = silhouette_score(new_data, hierarchical_labels)
print("Mean Silhouette width for Hierarchical Clustering:", silhouette_avg_hierarchical)

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(new_data)
new_data_with_clusters['dbscan_cluster'] = dbscan_labels

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='coolwarm',
alpha=0.6, s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('DBSCAN Clustering')
plt.show()

silhouette_avg_dbscan = silhouette_score(new_data, dbscan_labels)
print("Mean Silhouette Width for DBSCAN Clustering:",
silhouette_avg_dbscan)
