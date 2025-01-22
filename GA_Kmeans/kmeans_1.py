import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

# Helper function to calculate Euclidean distance
def calculate_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

# Manual KMeans clustering
def manual_kmeans_clustering(points, num_uavs, max_iter=300):
    np.random.seed(42)
    centroids = points[np.random.choice(points.shape[0], num_uavs, replace=False)]
    
    for iteration in range(max_iter):
        clusters = [[] for _ in range(num_uavs)]
        for point in points:
            distances = [calculate_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)

        new_centroids = np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else centroids[i] for i, cluster in enumerate(clusters)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Main function
if __name__ == "__main__":
    # Generate random waypoints
    num_points = 100
    num_uavs = 3
    np.random.seed(42)
    points = np.random.rand(num_points, 3) * 100

    # Perform KMeans clustering
    clusters, centroids = manual_kmeans_clustering(points, num_uavs)

    # Add the start point [0, 0, 0] to each cluster
    clusters_with_start = []
    for cluster in clusters:
        cluster_with_start = [np.array([0, 0, 0])] + cluster
        clusters_with_start.append(np.array(cluster_with_start))

    # Save the clusters with start points to a file
    with open("kmeans_output.pkl", "wb") as f:
        pickle.dump(clusters_with_start, f)
    print("KMeans output saved to 'kmeans_output.pkl'.")

    # Print the clusters
    for i, cluster in enumerate(clusters_with_start):
        print(f"Waypoints for UAV {i+1}:")
        print(cluster)
        print()

    # Plot the clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.rainbow(np.linspace(0, 1, num_uavs))
    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=colors[i], label=f"UAV {i+1} Cluster")

    ax.scatter(0, 0, 0, c="red", s=100, marker="o", label="Start Point")
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="black", s=100, marker="x", label="Centroids")
    ax.set_title(f"KMeans Clustering for {num_uavs} UAVs")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

    plot_kmeans_clusters(clusters, centroids, num_uavs)
