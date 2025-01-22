import numpy as np

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points in 3D space."""
    return np.linalg.norm(point1 - point2)

def initialize_centroids(points, num_uavs):
    """
    Initialize centroids using a density-based heuristic (similar to KMeans++).
    Start with a random point, then iteratively select the farthest point from existing centroids.
    """
    centroids = [points[np.random.choice(len(points))]]  # Start with a random point
    for _ in range(1, num_uavs):
        distances = np.array([min([calculate_distance(p, c) for c in centroids]) for p in points])
        new_centroid = points[np.argmax(distances)]
        centroids.append(new_centroid)
    return np.array(centroids)

def spatial_contiguity_adjustment(clusters, centroids):
    """
    Adjust clusters to ensure spatial contiguity. Assign outliers closer to neighboring centroids.
    """
    adjusted_clusters = [[] for _ in centroids]
    for i, cluster in enumerate(clusters):
        for point in cluster:
            # Find the closest centroid for each point
            distances = [calculate_distance(point, c) for c in centroids]
            closest_centroid_idx = np.argmin(distances)
            adjusted_clusters[closest_centroid_idx].append(point)
    return adjusted_clusters

def enhanced_kmeans(points, num_uavs, max_iter=300, balance_penalty=1.5):
    """
    Perform enhanced KMeans clustering with dynamic balancing and spatial adjustments.
    
    Args:
        points (np.ndarray): Array of points in 3D space.
        num_uavs (int): Number of UAVs (clusters).
        max_iter (int): Maximum number of iterations for convergence.
        balance_penalty (float): Factor to penalize unbalanced cluster sizes.

    Returns:
        clusters (list of np.ndarray): List of clusters with assigned points.
        centroids (np.ndarray): Final cluster centroids.
    """
    centroids = initialize_centroids(points, num_uavs)

    for iteration in range(max_iter):
        # Step 2: Assign points with weighted distance & dynamic balancing
        clusters = [[] for _ in range(num_uavs)]
        for point in points:
            distances = [
                calculate_distance(point, centroid) * (1 + len(cluster) / balance_penalty)
                for centroid, cluster in zip(centroids, clusters)
            ]
            clusters[np.argmin(distances)].append(point)

        # Convert clusters to numpy arrays
        clusters = [np.array(cluster) for cluster in clusters]

        # Step 3: Recalculate centroids
        new_centroids = [
            np.mean(cluster, axis=0) if len(cluster) > 0 else centroids[i]
            for i, cluster in enumerate(clusters)
        ]

        # Step 4: Spatial Contiguity Adjustment
        clusters = spatial_contiguity_adjustment(clusters, new_centroids)

        # Check convergence
        if np.allclose(new_centroids, centroids):
            print(f"Converged after {iteration+1} iterations")
            break
        centroids = new_centroids

    return clusters, np.array(centroids)

def save_kmeans_results(clusters, centroids, filename="kmeans_output.pkl"):
    """Save KMeans results to a file."""
    import pickle
    with open(filename, "wb") as f:
        pickle.dump({"clusters": clusters, "centroids": centroids}, f)
    print(f"KMeans results saved to {filename}")
