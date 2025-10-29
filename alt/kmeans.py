import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1️⃣ Create a self-made dataset
np.random.seed(42)
cluster1 = np.random.randn(50, 2) + np.array([2, 2])
cluster2 = np.random.randn(50, 2) + np.array([6, 6])
cluster3 = np.random.randn(50, 2) + np.array([2, 6])

X = np.vstack((cluster1, cluster2, cluster3))

# 2️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# 4️⃣ Get results
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster centers (scaled):\n", centroids)

# 5️⃣ Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title("K-Means Clustering on Self-Made Dataset")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()
plt.show()
