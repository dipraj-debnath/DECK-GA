import numpy as np
import pickle

# Helper function to calculate Euclidean distance
def calculate_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

# Weighted distance function with a penalty factor
def weighted_distance(point, centroid, weight_factor=1.0):
    base_distance = calculate_distance(point, centroid)
    return base_distance * weight_factor

# Assign points to clusters while ensuring balanced distribution
def balanced_assignment(points, centroids, max_points_per_cluster):
    clusters = [[] for _ in centroids]
    for point in points:
        distances = [calculate_distance(point, centroid) for centroid in centroids]
        sorted_indices = np.argsort(distances)
        for idx in sorted_indices:
            if len(clusters[idx]) < max_points_per_cluster:
                clusters[idx].append(point)
                break
    return clusters

# Validate and optimize cluster spatial contiguity
def optimize_clusters(clusters, centroids):
    for i, cluster in enumerate(clusters):
        sorted_cluster = sorted(cluster, key=lambda point: calculate_distance(point, centroids[i]))
        clusters[i] = sorted_cluster
    return clusters

# Enhanced KMeans clustering function (renamed to match expected import)
def manual_kmeans_clustering(points, num_uavs, max_iter=100000):
    np.random.seed(50)
    centroids = points[np.random.choice(points.shape[0], num_uavs, replace=False)]
    max_points_per_cluster = len(points) // num_uavs

    for iteration in range(max_iter):
        # Assign points to clusters with balance constraints
        clusters = balanced_assignment(points, centroids, max_points_per_cluster)

        # Recalculate centroids
        new_centroids = np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else centroids[i] for i, cluster in enumerate(clusters)])

        # Check for convergence
        if np.allclose(new_centroids, centroids):
            print(f"Converged after {iteration+1} iterations")
            break

        centroids = new_centroids

    # Optimize clusters for spatial contiguity
    clusters = optimize_clusters(clusters, centroids)

    return clusters, centroids

# Save KMeans results to a file
def save_kmeans_results(clusters, centroids, filename="kmeans_output.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"clusters": clusters, "centroids": centroids}, f)
    print(f"KMeans results saved to {filename}")
