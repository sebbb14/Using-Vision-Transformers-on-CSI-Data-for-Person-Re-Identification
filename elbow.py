
from plots import *
from sklearn.cluster import KMeans


def elbow_method_plot(signatures_normalized):
    wcss = []
    k_values = range(1, 10)  # Trying different cluster counts from 1 to 10

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(signatures_normalized)
        wcss.append(kmeans.inertia_)

    # Plotting the elbow curve
    plt.figure(figsize=(8, 6))  # Increase plot size
    plt.plot(k_values, wcss, marker='o', markersize=8)
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.ylim([min(wcss) * 0.9, max(wcss) * 1.1])  # Adjust y-axis limits
    plt.grid(True)
    plt.show()



