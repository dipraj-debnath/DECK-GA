import numpy as np
import pickle

# Function for Path Splitting Clustering
def path_splitting_clustering(points, num_uavs=3):
    """
    Perform path splitting clustering by dividing points into angular sectors.

    :param points: Array of 3D points (Nx3).
    :param num_uavs: Number of UAVs (clusters).
    :return: List of clusters (one for each UAV).
    """
    # Step 2: Define the rendezvous point (e.g., the centroid of all points)
    rendezvous_point = np.mean(points, axis=0)

    # Step 3: Calculate angles of points relative to the rendezvous point
    vector_from_rendezvous = points - rendezvous_point
    azimuth_angles = np.arctan2(vector_from_rendezvous[:, 1], vector_from_rendezvous[:, 0])
    elevation_angles = np.arctan2(
        vector_from_rendezvous[:, 2], np.sqrt(vector_from_rendezvous[:, 0] ** 2 + vector_from_rendezvous[:, 1] ** 2)
    )

    # Step 4: Combine the azimuth and elevation angles to get a single sorting key
    combined_angles = azimuth_angles + elevation_angles

    # Step 5: Sort points by the combined angle
    sorted_indices = np.argsort(combined_angles)
    sorted_points = points[sorted_indices]

    # Step 6: Divide the sorted points into equal angular sectors
    num_points = len(sorted_points)
    clusters = []
    for i in range(num_uavs):
        start_idx = i * num_points // num_uavs
        end_idx = (i + 1) * num_points // num_uavs
        clusters.append(sorted_points[start_idx:end_idx])

    return clusters, rendezvous_point

# Function to save the Path Splitting results
def save_path_splitting_results(clusters, rendezvous_point, filename="path_splitting_output.pkl"):
    """
    Save Path Splitting clustering results to a file.

    :param clusters: List of clusters.
    :param rendezvous_point: The rendezvous point used for clustering.
    :param filename: Name of the file to save results.
    """
    with open(filename, "wb") as f:
        pickle.dump({"clusters": clusters, "rendezvous_point": rendezvous_point}, f)
    print(f"Path Splitting results saved to {filename}")

# # Example of using the functions (can be removed for integration)
# if __name__ == "__main__":
#     np.random.seed(42)
#     points = np.random.rand(30, 3) * 100

#     clusters, rendezvous_point = path_splitting_clustering(points)
#     save_path_splitting_results(clusters, rendezvous_point)
