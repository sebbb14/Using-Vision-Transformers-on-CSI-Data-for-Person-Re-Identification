import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def kmeans_algorithm(pose_signatures, n_clusters):

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pose_signatures)

    # Reduce dimensions for visualization (optional, but helpful for high-dimensional data)
    pca = PCA(n_components=2)
    pose_signatures_2d = pca.fit_transform(pose_signatures)

    plt.figure(figsize=(10, 7))
    for cluster_id in range(n_clusters):
        cluster_points = pose_signatures_2d[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id + 1}')

    # Mark cluster centroids
    centroids_2d = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', s=200, marker='X', label='Centroids')

    # Add plot details
    plt.title("K-Means Clustering of Pose Signatures")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()