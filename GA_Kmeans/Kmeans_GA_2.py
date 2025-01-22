import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Kmeans_Updated_1 import manual_kmeans_clustering
from GA_path_planning_1 import ga_3d_pathplanning
import time

def main():
    # Generate random 3D points
    num_points = 100
    num_uavs = 3
    np.random.seed(42)
    points = np.random.rand(num_points, 3) * 100

    # Print the input points for KMeans
    print("\n--- KMeans Input ---")
    print("Points:")
    print(points)
    print(f"\nNumber of UAVs: {num_uavs}")

    # Run KMeans clustering
    start_time_kmeans = time.time()
    clusters, centroids = manual_kmeans_clustering(points, num_uavs)
    elapsed_time_kmeans = time.time() - start_time_kmeans

    # Print the KMeans output
    print("\n--- KMeans Output ---")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1} Waypoints:")
        print(np.array(cluster))
    print("\nCentroids:")
    print(centroids)
    print(f"\nKMeans clustering completed in {elapsed_time_kmeans:.2f} seconds.")

    # Plot KMeans clustering results
    fig_kmeans = plt.figure()
    ax_kmeans = fig_kmeans.add_subplot(111, projection="3d")
    ax_kmeans.set_title("KMeans Clustering")
    colors = cm.rainbow(np.linspace(0, 1, num_uavs))

    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        ax_kmeans.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=colors[i], label=f"Cluster {i+1}")
    ax_kmeans.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="black", s=100, marker="x", label="Centroids")
    ax_kmeans.set_xlabel("X")
    ax_kmeans.set_ylabel("Y")
    ax_kmeans.set_zlabel("Z")
    ax_kmeans.legend()
    plt.show()

    # Add start point (0, 0, 0) to each cluster
    clusters_with_start = [np.vstack([[0, 0, 0], np.array(cluster)]) for cluster in clusters]

    # Run GA optimization for each UAV
    optimized_paths = []
    start_time_ga = time.time()

    for i, cluster_points in enumerate(clusters_with_start):
        print(f"\n--- GA Input for UAV {i+1} ---")
        print("Waypoints:")
        print(cluster_points)
        optimized_path = ga_3d_pathplanning(cluster_points)
        optimized_paths.append(optimized_path)
        print(f"\n--- GA Output for UAV {i+1} ---")
        print("Optimized Route:")
        print(optimized_path)

    elapsed_time_ga = time.time() - start_time_ga
    print(f"\nGA optimization completed in {elapsed_time_ga:.2f} seconds.")

    # Plot the results for each UAV
    for i, path in enumerate(optimized_paths):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "o-", label=f"UAV {i+1}")
        ax.scatter(0, 0, 0, c="red", s=100, marker="o", label="Start/End Point")
        ax.set_title(f"Optimized Path for UAV {i+1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    # Combined plot for all UAVs
    fig_combined = plt.figure()
    ax_combined = fig_combined.add_subplot(111, projection="3d")
    colors = cm.rainbow(np.linspace(0, 1, len(optimized_paths)))

    for i, path in enumerate(optimized_paths):
        ax_combined.plot(path[:, 0], path[:, 1], path[:, 2], label=f"UAV {i+1}", color=colors[i])
    ax_combined.scatter(0, 0, 0, c="red", s=100, marker="o", label="Start/End Point")
    ax_combined.set_title("Optimized UAV Paths")
    ax_combined.set_xlabel("X")
    ax_combined.set_ylabel("Y")
    ax_combined.set_zlabel("Z")
    ax_combined.legend()
    plt.show()

    # Summary of timing
    print(f"\n--- Summary ---")
    print(f"KMeans Time: {elapsed_time_kmeans:.2f} seconds")
    print(f"GA Time: {elapsed_time_ga:.2f} seconds")
    print(f"Total Time: {elapsed_time_kmeans + elapsed_time_ga:.2f} seconds")

if __name__ == "__main__":
    main()
