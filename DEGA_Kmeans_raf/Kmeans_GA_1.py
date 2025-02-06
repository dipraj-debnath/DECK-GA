import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Kmeans_2 import manual_kmeans_clustering, save_kmeans_results
from GA_path_planning import ga_3d_pathplanning

# Main function to run KMeans and GA
def main():
    # Step 1: Generate random 3D waypoints
    num_points = 100
    num_uavs = 3
    np.random.seed(42)
    points = np.random.rand(num_points, 3) * 100

    # Step 2: Run KMeans clustering
    clusters, centroids = manual_kmeans_clustering(points, num_uavs)
    save_kmeans_results(clusters, centroids)

    # Add start point [0, 0, 0] to each cluster
    clusters_with_start = [np.vstack([[0, 0, 0], np.array(cluster)]) for cluster in clusters]

    # Step 3: Run GA for each UAV and plot
    optimized_paths = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.rainbow(np.linspace(0, 1, len(clusters_with_start)))

    for i, cluster_points in enumerate(clusters_with_start):
        optimized_path = ga_3d_pathplanning(cluster_points)
        optimized_paths.append(optimized_path)

        # Plot individual UAV path
        ax.plot(optimized_path[:, 0], optimized_path[:, 1], optimized_path[:, 2], color=colors[i], label=f"UAV {i+1}")
    
    ax.scatter(0, 0, 0, c="red", s=100, marker="o", label="Start/End Point")
    ax.set_title("Optimized UAV Paths")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

    print("\n--- Optimized Paths for All UAVs ---")
    for i, path in enumerate(optimized_paths):
        print(f"UAV {i+1} Path:")
        print(path)

if __name__ == "__main__":
    main()
