import numpy as np
import pickle

def dynamic_assignment_clustering(points, num_uavs):
    """
    Perform Dynamic Assignment clustering on 3D points using UAV movement.

    :param points: Array of 3D points.
    :param num_uavs: Number of UAVs/clusters.
    :return: List of clusters and final UAV positions (centroids).
    """
    # Initialize UAV positions (all starting from the origin)
    uav_positions = np.zeros((num_uavs, 3))

    # Initialize lists to keep track of assigned points for each UAV
    clusters = [[] for _ in range(num_uavs)]
    remaining_points = points.copy()

    # Dynamically assign points to UAVs
    while len(remaining_points) > 0:
        for uav in range(num_uavs):
            if len(remaining_points) == 0:
                break

            # Find the nearest point to the current UAV position
            distances = np.linalg.norm(remaining_points - uav_positions[uav], axis=1)
            nearest_point_index = np.argmin(distances)
            nearest_point = remaining_points[nearest_point_index]

            # Assign the nearest point to the UAV
            clusters[uav].append(nearest_point)

            # Update the UAV position to the newly assigned point
            uav_positions[uav] = nearest_point

            # Remove the assigned point from the remaining points
            remaining_points = np.delete(remaining_points, nearest_point_index, axis=0)

    # Convert clusters to numpy arrays
    clusters = [np.array(cluster) for cluster in clusters]

    return clusters, uav_positions

def save_dynamic_assignment_results(clusters, centroids, filename="dynamic_assignment_output.pkl"):
    """
    Save the Dynamic Assignment clustering results to a file.

    :param clusters: List of clusters.
    :param centroids: Array of final UAV positions.
    :param filename: Name of the file to save the results.
    """
    with open(filename, "wb") as f:
        pickle.dump({"clusters": clusters, "centroids": centroids}, f)
    print(f"Dynamic Assignment results saved to {filename}")
