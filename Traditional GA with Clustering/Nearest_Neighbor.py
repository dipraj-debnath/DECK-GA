import numpy as np
import pickle

def nearest_neighbor_clustering(points, num_uavs=3, start_positions=None):
    """
    Perform clustering using the Nearest Neighbor heuristic.

    :param points: Array of 3D waypoints.
    :param num_uavs: Number of UAVs (clusters).
    :param start_positions: Starting positions of UAVs, default to origin.
    :return: A tuple (clusters, centroids).
    """
    if start_positions is None:
        start_positions = np.zeros((num_uavs, 3))  # Default start at (0, 0, 0)

    remaining_points = points.copy()
    clusters = [[] for _ in range(num_uavs)]
    current_positions = start_positions.copy()

    while len(remaining_points) > 0:
        for uav in range(num_uavs):
            if len(remaining_points) == 0:
                break
            # Find the nearest point to the current UAV position
            distances = np.linalg.norm(remaining_points - current_positions[uav], axis=1)
            nearest_point_index = np.argmin(distances)
            nearest_point = remaining_points[nearest_point_index]

            # Assign the nearest point to the UAV
            clusters[uav].append(nearest_point)

            # Update the current UAV position
            current_positions[uav] = nearest_point

            # Remove the assigned point
            remaining_points = np.delete(remaining_points, nearest_point_index, axis=0)

    centroids = np.array([np.mean(cluster, axis=0) if cluster else np.zeros(3) for cluster in clusters])
    return [np.array(cluster) for cluster in clusters], centroids

def save_nearest_neighbor_results(clusters, centroids, filename="nearest_neighbor_output.pkl"):
    """
    Save Nearest Neighbor clustering results to a file.

    :param clusters: Clusters of points.
    :param centroids: Centroids of clusters.
    :param filename: File to save the results.
    """
    with open(filename, "wb") as f:
        pickle.dump({"clusters": clusters, "centroids": centroids}, f)
    print(f"Nearest Neighbor results saved to {filename}")
