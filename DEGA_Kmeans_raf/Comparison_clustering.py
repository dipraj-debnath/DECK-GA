# from kmeans_Updated_3 import manual_kmeans_clustering, save_kmeans_results
from Kmeans_2 import manual_kmeans_clustering, save_kmeans_results
from GA_path_planning import ga_3d_pathplanning
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import pickle

# Load the saved points
def load_points(filename="points.pkl"):
    """
    Load 3D waypoints from a file.

    :param filename: Name of the file to load points from.
    :return: Array of 3D points.
    """
    with open(filename, "rb") as f:
        points = pickle.load(f)
    print(f"Loaded points from {filename}")
    return points

    
# Function to calculate total path distance
def calculate_path_distance(path):
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))


# Main function to run KMeans and GA
def main():
    # Step 1: Generate random 3D waypoints
    #num_points = 100
    #num_uavs = 3
    #np.random.seed(35)
    #points = np.random.rand(num_points, 3) * 100
    # Step 1: Load 3D waypoints
    points = load_points()
    num_points = len(points)
    num_uavs = 3
    plotting = False
    # Print the input points for KMeans
    print("\n--- KMeans Input ---")
    print("Points:")
    print(points)
    print(f"\nNumber of UAVs: {num_uavs}")

    # Step 2: Run KMeans clustering
    start_time_kmeans = time.time()
    clusters, centroids = manual_kmeans_clustering(points, num_uavs)
    elapsed_time_kmeans = time.time() - start_time_kmeans

    # Save KMeans results
    save_kmeans_results(clusters, centroids)

    # Print KMeans results
    print("\n--- KMeans Output ---")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1} Waypoints:")
        print(np.array(cluster))
    print("\nCentroids:")
    print(centroids)
    print(f"KMeans clustering completed in {elapsed_time_kmeans:.2f} seconds.")

    # Step 3: Run GA for each UAV

    start_points = [[0, 0, 0], [10, 10, 10], [50, 50, 20]]
    clusters_with_start = [np.vstack([start_points[i], np.array(cluster)]) for i, cluster in enumerate(clusters)]

    #clusters_with_start = [np.vstack([[0, 0, 0], np.array(cluster)]) for cluster in clusters]

    # Calculate total path distances before optimization
    total_distance_before = [calculate_path_distance(cluster) for cluster in clusters_with_start]

    print("\n--- Total Path Distances Before Optimization ---")
    for i, distance in enumerate(total_distance_before):
        print(f"UAV {i+1}: {distance:.2f}")

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

    # Calculate total path distances after optimization
    total_distance_after = [calculate_path_distance(path) for path in optimized_paths]

    print("\n--- Total Path Distances After Optimization ---")
    for i, distance in enumerate(total_distance_after):
        print(f"UAV {i+1}: {distance:.2f}")

    # Calculate and print distance reduction
    print("\n--- Distance Reduction ---")
    for i in range(num_uavs):
        reduction = total_distance_before[i] - total_distance_after[i]
        print(f"UAV {i+1}: {reduction:.2f} (Reduced by {reduction / total_distance_before[i] * 100:.2f}%)")

    # Summary of timing
    print(f"\n--- Summary ---")
    print(f"KMeans Time: {elapsed_time_kmeans:.2f} seconds")
    print(f"GA Time: {elapsed_time_ga:.2f} seconds")
    print(f"Total Time: {elapsed_time_kmeans + elapsed_time_ga:.2f} seconds")
    
    #now for simulated annealing
    
    # Step 2: Run KMeans clustering
    start_time_kmeans = time.time()
    clusters, centroids = manual_kmeans_clustering(points, num_uavs)
    elapsed_time_kmeans = time.time() - start_time_kmeans

    # Save KMeans results
    save_kmeans_results(clusters, centroids)

    # Print KMeans results
    print("\n--- KMeans Output ---")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1} Waypoints:")
        print(np.array(cluster))
    print("\nCentroids:")
    print(centroids)
    print(f"KMeans clustering completed in {elapsed_time_kmeans:.2f} seconds.")

    # Step 3: Run GA for each UAV

    start_points = [[0, 0, 0], [10, 10, 10], [50, 50, 20]]
    clusters_with_start = [np.vstack([start_points[i], np.array(cluster)]) for i, cluster in enumerate(clusters)]

    #clusters_with_start = [np.vstack([[0, 0, 0], np.array(cluster)]) for cluster in clusters]

    # Calculate total path distances before optimization
    total_distance_before = [calculate_path_distance(cluster) for cluster in clusters_with_start]

    print("\n--- Total Path Distances Before Optimization ---")
    for i, distance in enumerate(total_distance_before):
        print(f"UAV {i+1}: {distance:.2f}")

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

    # Calculate total path distances after optimization
    total_distance_after = [calculate_path_distance(path) for path in optimized_paths]

    print("\n--- Total Path Distances After Optimization ---")
    for i, distance in enumerate(total_distance_after):
        print(f"UAV {i+1}: {distance:.2f}")

    # Calculate and print distance reduction
    print("\n--- Distance Reduction ---")
    for i in range(num_uavs):
        reduction = total_distance_before[i] - total_distance_after[i]
        print(f"UAV {i+1}: {reduction:.2f} (Reduced by {reduction / total_distance_before[i] * 100:.2f}%)")

    # Summary of timing
    print(f"\n--- Summary ---")
    print(f"KMeans Time: {elapsed_time_kmeans:.2f} seconds")
    print(f"GA Time: {elapsed_time_ga:.2f} seconds")
    print(f"Total Time: {elapsed_time_kmeans + elapsed_time_ga:.2f} seconds")

if __name__ == "__main__":
    main()
