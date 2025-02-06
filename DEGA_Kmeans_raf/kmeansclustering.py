import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Helper function to calculate Euclidean distance (Pythagoras)
def calculate_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

# Manual KMeans clustering using Pythagorean distance formula
def manual_kmeans_clustering(points, num_uavs, max_iter=300):
    # Step 1: Randomly initialize centroids
    np.random.seed(42)
    centroids = points[np.random.choice(points.shape[0], num_uavs, replace=False)]
    
    for iteration in range(max_iter):
        # Step 2: Assign points to the nearest centroid
        clusters = [[] for _ in range(num_uavs)]  # Empty lists for each cluster

        for point in points:
            distances = [calculate_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)

        # Step 3: Recompute centroids
        new_centroids = np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else centroids[i] for i, cluster in enumerate(clusters)])

        # Step 4: Check for convergence (if centroids do not change)
        if np.allclose(new_centroids, centroids):
            print(f"Converged after {iteration+1} iterations")
            break

        centroids = new_centroids  # Update centroids for next iteration

    return clusters, centroids

# Plotting the manually computed KMeans clusters
def plot_manual_kmeans(points, clusters, centroids, num_uavs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Manual KMeans Clustering with {num_uavs} UAVs')

    colors = cm.rainbow(np.linspace(0, 1, num_uavs))

    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=colors[i], label=f'UAV {i+1}')
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', s=100, marker='x', label='Centroids')

    # Plot start point
    ax.scatter(0, 0, 0, c='purple', s=100, marker='o', label='Start Point')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Generate random points and run manual KMeans
def run_manual_kmeans_clustering(num_points=100, num_uavs=3):
    # Generate random 3D points
    np.random.seed(42)
    points = np.random.rand(num_points, 3) * 100

    # Run manual KMeans clustering
    clusters, centroids = manual_kmeans_clustering(points, num_uavs)

    # Plot the results
    plot_manual_kmeans(points, clusters, centroids, num_uavs)

    return clusters, centroids

# Example usage
clusters, centroids = run_manual_kmeans_clustering(num_points=100, num_uavs=3)

# Print out the clusters and centroids
for i, cluster in enumerate(clusters):
    print(f"\nUAV {i+1} has {len(cluster)} points.")
    print(f"Centroid of UAV {i+1} is at {centroids[i]}")

