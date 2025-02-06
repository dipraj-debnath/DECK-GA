from Simulated_Annealing import simulated_annealing_clustering  # Import Simulated Annealing function
from Basic_GA import basic_ga_path_planning  # Import Basic GA function
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time


def load_points(filename="points_100).pkl"):
    """
    Load 3D waypoints from a file.

    :param filename: Name of the file to load points from.
    :return: Array of 3D points.
    """
    with open(filename, "rb") as f:
        points = pickle.load(f)
    print(f"Loaded points from {filename}")
    return points


def calculate_total_distance(path):
    """
    Calculate the total distance of a path.

    :param path: A 2D numpy array representing the path.
    :return: Total distance of the path.
    """
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))


def main():
    # Step 1: Load 3D waypoints
    points = load_points("points_100.pkl")  # Change this to your file (e.g., points100.pkl or points30.pkl)
    num_uavs = 3

    print("\n--- Simulated Annealing Input ---")
    print(f"Points:\n{points}")
    print(f"Number of UAVs: {num_uavs}")

    # Step 2: Run Simulated Annealing clustering
    start_time_sa = time.time()  # Start timing Simulated Annealing
    clusters, centroids = simulated_annealing_clustering(points, num_uavs)
    elapsed_time_sa = time.time() - start_time_sa  # End timing Simulated Annealing

    print("\n--- Simulated Annealing Output ---")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1} Waypoints:\n{np.array(cluster)}")
    print(f"\nCentroids:\n{centroids}")
    print(f"Simulated Annealing clustering completed in {elapsed_time_sa:.2f} seconds.")

    # Step 3: Run Basic GA for each UAV
    # start_points = [[10, 0, 0], [40, 40, 0], [80, 0, 0]]  # Fixed start points for each UAV
    start_points = [[10, 0, 0], [50, 20, 0], [90, 0, 0]]
    clusters_with_start = [
        np.vstack([start_points[i], np.array(cluster)]) for i, cluster in enumerate(clusters)
    ]

    optimized_paths = []
    total_distances = []

    start_time_ga = time.time()  # Start timing GA
    for i, cluster_points in enumerate(clusters_with_start):
        print(f"\n--- Running Basic GA for UAV {i + 1} ---")

        # Ensure the cluster is non-empty
        if len(cluster_points) == 1:  # Only contains the start point
            print(f"Warning: Cluster {i + 1} has no waypoints. Skipping GA for this UAV.")
            continue

        # Dynamically adjust population size if the cluster has fewer points
        population_size = min(50, len(cluster_points))  # Replace 50 with your desired default population size
        num_iterations = 25000  # Define the number of iterations for GA
        optimized_path = basic_ga_path_planning(cluster_points, population_size=population_size, num_iterations=num_iterations)

        # Ensure UAV returns to the starting point
        optimized_path = np.vstack([optimized_path, optimized_path[0]])

        optimized_paths.append(optimized_path)
        total_distance = calculate_total_distance(optimized_path)
        total_distances.append(total_distance)

        print(f"Optimized Path for UAV {i + 1}:\n{optimized_path}")
        print(f"Total Distance for UAV {i + 1}: {total_distance:.2f}")

    elapsed_time_ga = time.time() - start_time_ga  # End timing GA

    # Step 4: Calculate total combined distance
    combined_distance = sum(total_distances)
    print(f"\n--- Total Combined Distance for All UAVs: {combined_distance:.2f} ---")

    # # Step 5: Plot the results
    # fig_combined = plt.figure(figsize=(8, 6))
    # ax_combined = fig_combined.add_subplot(111, projection="3d")
    # colors = cm.rainbow(np.linspace(0, 1, num_uavs))

    # for i, path in enumerate(optimized_paths):
    #     ax_combined.plot(path[:, 0], path[:, 1], path[:, 2], '*-', label=f"UAV {i + 1} Path", color=colors[i])
    #     ax_combined.scatter(start_points[i][0], start_points[i][1], start_points[i][2], c="red", s=100, marker="o", label=f"UAV {i + 1} Start")

    # ax_combined.set_title("Optimized Paths for UAVs")
    # ax_combined.set_xlabel("X")
    # ax_combined.set_ylabel("Y")
    # ax_combined.set_zlabel("Z")
    # ax_combined.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    # plt.tight_layout()
    # plt.show()

    # Print the timing summary
    print(f"\n--- Timing Summary ---")
    print(f"Simulated Annealing Time: {elapsed_time_sa:.2f} seconds")
    print(f"GA Time: {elapsed_time_ga:.2f} seconds")
    print(f"Total Time: {elapsed_time_sa + elapsed_time_ga:.2f} seconds")


if __name__ == "__main__":
    main()
