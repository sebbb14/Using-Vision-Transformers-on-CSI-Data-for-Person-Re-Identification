import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap
from tensorflow.python.ops.nn_impl import l2_normalize
import torch

def pca(data, n_components):
    pca = PCA(n_components=n_components)
    reduced_vector = pca.fit_transform(data)

    return reduced_vector

def t_sne(data, n_components):
    tsne = TSNE(n_components=n_components, perplexity=29, random_state=42)
    reduced_vector = tsne.fit_transform(data)

    return reduced_vector

def umap_reduction(data, n_components):

    reducer = umap.UMAP(n_components=n_components)  # Set desired dimensionality
    reduced_data = reducer.fit_transform(data)  # Returns a 2D array with shape (n_samples, 2)

    return reduced_data

def normalize_L2(data):

    l2_norms = np.linalg.norm(data, axis=1, keepdims=True)

    # Normalize the vectors, handling the case where norm is 0 to avoid division by zero
    normalized_vectors = np.divide(data, l2_norms, out=np.zeros_like(data, dtype=float), where=l2_norms != 0)

    return normalized_vectors


def plot_elbow_method(data):

    # number of cluster of example for the elbow method
    k_values = range(1, 11)
    inertia_values = []

    # Calculate inertia for each value of k
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)

    # Plot the elbow method
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertia_values, marker='o', linestyle='-', label='Inertia')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14)
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()


# Example usage with a random dataset
if __name__ == "__main__":

    data = np.load("./saved_features_concatenated/epoch_100_embeddings.npy")


    combined = torch.load("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/combined/checkpoints_199.pt")
    data = combined["embeddings"].detach().numpy()
    last_labels = combined["labels"].detach().numpy()
    print(last_labels)
    print(data.shape)


    # new embeddings calculated after training
    data = torch.load("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/all_embeddings.pt")
    data = np.array([x.squeeze() for x in data])
    # data = np.load("./saved_features_concatenated/epoch_100_embeddings.npy")
    # data = torch.load("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/.pt")

    last_labels = np.load("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/labels.npy")
    print(last_labels)
    print(data.shape)

    # save the labels of the last embeddgings
    # last_labels = np.load("labels.npy")
    unique_names = np.unique(last_labels)
    # Create a dictionary mapping names to integers
    name_to_int = {name: idx for idx, name in enumerate(unique_names)}
    # Convert the original list to integers -> [0,0,0,..., 4,4,4,...]
    last_labels = np.array([name_to_int[name] for name in last_labels])
    # last_labels = torch.tensor(labels)

    data = l2_normalize(data)
    data = t_sne(data, 2)


    # Call the elbow method plot function
    plot_elbow_method(data)

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)

    # Get cluster centers and labels
    centers = kmeans.cluster_centers_
    # labels = kmeans.labels_

    # Visualize the clusters
    for i in range(k):
         cluster_points = data[last_labels == i]
         plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

    # Annotate points with labels (if you want to use the last_labels array for annotation)
    for idx, point in enumerate(data):
        plt.text(point[0], point[1], str(last_labels[idx]), fontsize=12, color='black')

    plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=100, label='Centroids')
    plt.legend()
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Compute silhouette score
    silhouette_avg = silhouette_score(data, last_labels)
    print(f"Silhouette Score for {k} clusters: {silhouette_avg:.3f}")