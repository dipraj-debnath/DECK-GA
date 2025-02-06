import numpy as np
import pickle

def calculate_total_distance(cluster, centroid):
    """
    Calculate the total distance of points in a cluster from the centroid.
    """
    return np.sum(np.linalg.norm(np.array(cluster) - centroid, axis=1))

def manual_kmeans_clustering(points, num_clusters, max_iterations=100):
    """
    Perform KMeans clustering manually with additional balancing and optimization.
    """
    points = np.array(points)
    num_points = points.shape[0]

    # Step 1: Initialize centroids randomly
    np.random.seed(35)
    initial_indices = np.random.choice(num_points, num_clusters, replace=False)
    centroids = points[initial_indices]

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(num_clusters)]

        # Step 2: Assign points to the nearest centroid
        for point in points:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        # Step 3: Balance cluster sizes by moving excess points
        cluster_sizes = [len(cluster) for cluster in clusters]
        target_size = num_points // num_clusters

        for i in range(num_clusters):
            while len(clusters[i]) > target_size:
                # Identify the farthest point to move
                distances = [np.linalg.norm(point - centroids[i]) for point in clusters[i]]
                farthest_point_index = np.argmax(distances)
                point_to_move = clusters[i][farthest_point_index]

                # Find the nearest centroid that has space
                distances_to_other_centroids = [
                    np.linalg.norm(point_to_move - centroids[j]) if j != i else float('inf')
                    for j in range(num_clusters)
                ]
                target_cluster_index = np.argmin(distances_to_other_centroids)

                # Move the point
                for j, point in enumerate(clusters[i]):
                    if np.array_equal(point, point_to_move):
                        del clusters[i][j]
                        break
                clusters[target_cluster_index].append(point_to_move)

        # Step 4: Recalculate centroids
        new_centroids = np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else centroids[i]
                                   for i, cluster in enumerate(clusters)])

        # Step 5: Check for convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break

        centroids = new_centroids

    return clusters, centroids


def save_kmeans_results(clusters, centroids, filename="kmeans_output.pkl"):
    """
    Save KMeans results to a file.
    """
    with open(filename, "wb") as f:
        pickle.dump({"clusters": clusters, "centroids": centroids}, f)
    print(f"KMeans results saved to {filename}")
