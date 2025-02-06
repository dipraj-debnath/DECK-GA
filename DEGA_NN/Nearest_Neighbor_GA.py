import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from Nearest_Neighbor import nearest_neighbor_clustering, save_nearest_neighbor_results
from GA_path_planning import ga_3d_pathplanning
import pickle
import matplotlib
matplotlib.rc('font', family='sans-serif')  # Set the font to sans-serif


# Load the saved points
def load_points(filename="points_100.pkl"):
    with open(filename, "rb") as f:
        points = pickle.load(f)
    print(f"Loaded points from {filename}")
    return points

# Calculate total path distance
def calculate_path_distance(path):
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

def main():
    # Step 1: Load points and set parameters
    points = load_points()
    num_uavs = 3
    start_positions = np.array([[10, 0, 0], [50, 20, 0], [90, 0, 0]])

    # Step 2: Run Nearest Neighbor clustering
    start_time_nn = time.time()
    clusters, centroids = nearest_neighbor_clustering(points, num_uavs, start_positions)
    elapsed_time_nn = time.time() - start_time_nn

    # Save clustering results
    save_nearest_neighbor_results(clusters, centroids)

    # Print clustering results
    print("\n--- Nearest Neighbor Output ---")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1} Waypoints:")
        print(cluster)
    print("\nCentroids:")
    print(centroids)
    print(f"Nearest Neighbor clustering completed in {elapsed_time_nn:.2f} seconds.")

    # Step 3: Prepare clusters for GA with start positions
    clusters_with_start = [np.vstack([start_positions[i], cluster]) for i, cluster in enumerate(clusters)]

    # Calculate total path distances before optimization
    total_distance_before = [calculate_path_distance(cluster) for cluster in clusters_with_start]
    print("\n--- Total Path Distances Before Optimization ---")
    for i, distance in enumerate(total_distance_before):
        print(f"UAV {i+1}: {distance:.2f}")

    total_before = sum(total_distance_before)
    print(f"\nTotal Distance Before Optimization: {total_before:.2f}")

    # Step 4: Run GA for each UAV
    optimized_paths = []
    start_time_ga = time.time()
    for i, cluster in enumerate(clusters_with_start):
        optimized_path = ga_3d_pathplanning(cluster)
        optimized_paths.append(optimized_path)
    elapsed_time_ga = time.time() - start_time_ga

    # Calculate total path distances after optimization
    total_distance_after = [calculate_path_distance(path) for path in optimized_paths]
    print("\n--- Total Path Distances After Optimization ---")
    for i, distance in enumerate(total_distance_after):
        print(f"UAV {i+1}: {distance:.2f}")

    total_after = sum(total_distance_after)
    print(f"\nTotal Distance After Optimization: {total_after:.2f}")

    # Calculate and print distance reduction
    print("\n--- Distance Reduction ---")
    for i in range(num_uavs):
        reduction = total_distance_before[i] - total_distance_after[i]
        print(f"UAV {i+1}: {reduction:.2f} (Reduced by {reduction / total_distance_before[i] * 100:.2f}%)")

    # Calculate and print total path distances
    total_distance_before_sum = sum(total_distance_before)
    total_distance_after_sum = sum(total_distance_after)
    total_distance_reduction_sum = total_distance_before_sum - total_distance_after_sum

     # Print total distances
    print(f"\n--- Total Path Distances ---")
    print(f"Before Optimization: {total_distance_before_sum:.2f}")
    print(f"After Optimization: {total_distance_after_sum:.2f}")
    print(f"Total Distance Reduction: {total_distance_reduction_sum:.2f}")

    # Calculate total distance reduction percentage
    total_reduction_percentage = (total_distance_reduction_sum / total_distance_before_sum) * 100

    # Print total distance reduction with percentage
    print(f"\n--- Total Distance Reduction ---")
    print(f"Total Distance Reduction: {total_distance_reduction_sum:.2f} (Reduced by {total_reduction_percentage:.2f}%)")

    print(f"\nGA optimization completed in {elapsed_time_ga:.2f} seconds.")

    # Step 5: Plot results
    fig_nn = plt.figure()
    ax_nn = fig_nn.add_subplot(111, projection="3d")
    colors = cm.rainbow(np.linspace(0, 1, num_uavs))

    # # Plot clusters
    # for i, cluster in enumerate(clusters):
    #     cluster_points = np.array(cluster)
    #     ax_nn.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=colors[i], label=f"Cluster {i+1}")
    # ax_nn.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="black", s=100, marker="x", label="Centroids")
    # ax_nn.set_title("Nearest Neighbor Clustering")
    # ax_nn.set_xlabel("X")
    # ax_nn.set_ylabel("Y")
    # ax_nn.set_zlabel("Z")
    # ax_nn.legend()
    # plt.show()

    # # Plot optimized paths for each UAV
    # for i, path in enumerate(optimized_paths):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.plot(path[:, 0], path[:, 1], path[:, 2], "*-", label=f"UAV {i+1}")
    #     ax.scatter(start_positions[i][0], start_positions[i][1], start_positions[i][2], c="red", s=100, marker="o", label="Start Point")
    #     ax.set_title(f"Optimized Path for UAV {i+1}")
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.legend()
    #     plt.show()

    # # Combined plot for all UAVs
    # fig_combined = plt.figure(figsize=(3.5,2.5), dpi= 200)
    # ax_combined = fig_combined.add_subplot(111, projection="3d")
    # for i, path in enumerate(optimized_paths):
    #     ax_combined.plot(path[:, 0], path[:, 1], path[:, 2], "*-", label=f"UAV {i+1}", color=colors[i])
    # for i, start in enumerate(start_positions):
    #     ax_combined.scatter(*start, c="red", s=100, marker="o", label=f"UAV {i + 1} Start")   
    # ax_combined.set_title("Optimised UAV Paths", fontsize=9)
    # ax_combined.set_xlabel("X", fontsize=9)
    # ax_combined.set_ylabel("Y", fontsize=9)
    # ax_combined.set_zlabel("Z", fontsize=9)
    # #ax_combined.legend()
    # plt.show()

    # Summary of timing
    print(f"\n--- Summary ---")
    print(f"Nearest Neighbor Time: {elapsed_time_nn:.2f} seconds")
    print(f"GA Time: {elapsed_time_ga:.2f} seconds")
    print(f"Total Time: {elapsed_time_nn + elapsed_time_ga:.2f} seconds")

if __name__ == "__main__":
    main()
