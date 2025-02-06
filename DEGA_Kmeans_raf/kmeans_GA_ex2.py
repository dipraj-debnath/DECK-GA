import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# GA Path Planning function
def ga_3d_pathplanning(xyz):
    # Initialize the variables
    N = xyz.shape[0]  # Number of points
    popSize = 8       # Population size
    numIter = int(1e5)  # Number of iterations

    # Generate distance matrix in 3D
    a = np.meshgrid(np.arange(N), np.arange(N))
    dmat = np.round(np.sqrt(np.sum((xyz[a[0], :] - xyz[a[1], :])**2, axis=2)))

    # Sanity checks for population size
    popSize = 4 * int(np.ceil(popSize / 4))
    numIter = max(1, round(numIter))

    # Initialize population
    pop = np.zeros((popSize, N), dtype=int)
    pop[0, :] = np.arange(N)  # Start with sequential route
    for k in range(1, popSize):
        pop[k, :] = np.random.permutation(N)  # Random routes for the population

    # GA variables
    globalMin = np.inf
    distHistory = np.zeros(numIter)

    for iter in range(numIter):
        totalDist = np.zeros(popSize)

        # Calculate total distance for each member of the population
        for p in range(popSize):
            d = dmat[pop[p, -1], pop[p, 0]]  # Closed path
            for k in range(1, N):
                d += dmat[pop[p, k - 1], pop[p, k]]
            totalDist[p] = d

        minDist = np.min(totalDist)
        index = np.argmin(totalDist)
        distHistory[iter] = minDist

        if minDist < globalMin:
            globalMin = minDist
            optRoute = pop[index, :]

        # Genetic Algorithm Operators: mutation and crossover
        randomOrder = np.random.permutation(popSize)
        newPop = np.zeros((popSize, N), dtype=int)
        for p in range(0, popSize, 4):
            rtes = pop[randomOrder[p:p+4], :]
            dists = totalDist[randomOrder[p:p+4]]
            idx = np.argmin(dists)
            bestOf4Route = rtes[idx, :]
            routeInsertionPoints = np.sort(np.random.randint(0, N, size=2))
            I, J = routeInsertionPoints

            # Mutation operations (flip, swap, slide)
            tmpPop = np.zeros((4, N), dtype=int)
            tmpPop[0, :] = bestOf4Route

            # Flip mutation
            if I < J:
                tmpPop[1, I:J+1] = bestOf4Route[I:J+1][::-1]
            elif I > J:
                tmpPop[1, J:I+1] = bestOf4Route[J:I+1][::-1]
            else:
                tmpPop[1, :] = bestOf4Route

            # Swap mutation
            tmpPop[2, :] = bestOf4Route.copy()
            tmpPop[2, [I, J]] = bestOf4Route[[J, I]]

            # Slide mutation
            tmpPop[3, :] = bestOf4Route.copy()
            tmpPop[3, I:J+1] = np.roll(bestOf4Route[I:J+1], shift=-1)

            # Ensure population integrity: No duplicates and valid route
            for i in range(4):
                if len(np.unique(tmpPop[i, :])) != N:
                    tmpPop[i, :] = np.random.permutation(N)

            newPop[p:p+4, :] = tmpPop
        pop = newPop  # Update the population

    # Final optimized route
    final_route = xyz[optRoute, :]
    start_point = np.where(np.all(final_route == xyz[0], axis=1))[0][0]
    rearranged_route = np.concatenate((final_route[start_point:], final_route[:start_point+1]))

    return rearranged_route

# KMeans clustering and running GA for each UAV's cluster
def kmeans_clustering_3d(num_points=100, num_uavs=3):
    # Step 1: Generate random 3D points
    np.random.seed(42)  # For reproducibility
    points = np.random.rand(num_points, 3) * 100  # Random points in a 100x100x100 cube

    # Step 2: Initialize KMeans for the specified number of UAVs (clusters)
    kmeans = KMeans(n_clusters=num_uavs, random_state=42)
    kmeans.fit(points)

    # Step 3: Get the cluster assignments for each point
    labels = kmeans.labels_

    # Step 4: Store points for each UAV cluster
    uav_clusters = {}
    for i in range(num_uavs):
        cluster_points = points[labels == i]
        # Add the start point (0, 0, 0) to each UAV's points
        cluster_points = np.vstack([[0, 0, 0], cluster_points])
        uav_clusters[i] = cluster_points  # Save the points for each UAV

    return uav_clusters, points, labels  # Return the clusters and points for plotting

# Plot KMeans Output
def plot_kmeans_output(points, labels, num_uavs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'KMeans Clustering Output for {num_uavs} UAVs')

    colors = cm.rainbow(np.linspace(0, 1, num_uavs))  # Different color for each UAV

    # Plot KMeans output
    for i in range(num_uavs):
        cluster_points = points[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=colors[i], label=f'UAV {i+1}')

    # Plot start point
    ax.scatter(0, 0, 0, c='purple', s=100, marker='o', label='Start Point')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Solve paths for UAVs using GA and plot in the same figure
def solve_uav_paths_with_ga(num_points=100, num_uavs=3):
    # Step 1: Run K-Means clustering to divide points among UAVs
    uav_clusters, points, labels = kmeans_clustering_3d(num_points=num_points, num_uavs=num_uavs)

    # Step 2: Plot KMeans Output before GA
    plot_kmeans_output(points, labels, num_uavs)

    # Step 3: Run GA for each UAV's set of points and plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Optimized UAV Paths for {num_uavs} UAVs')

    colors = cm.rainbow(np.linspace(0, 1, num_uavs))  # Different color for each UAV

    optimized_routes = {}
    for uav_id, uav_points in uav_clusters.items():
        print(f"\nUAV {uav_id+1} Points (Including Start Point):\n", uav_points)
        
        # Run GA to get optimized route
        optimized_route = ga_3d_pathplanning(uav_points)
        optimized_routes[uav_id] = optimized_route

        # Plot the optimized route
        ax.plot(optimized_route[:, 0], optimized_route[:, 1], optimized_route[:, 2], 'o-', color=colors[uav_id], label=f'UAV {uav_id+1}')

    # Plot start and end point
    ax.scatter(0, 0, 0, c='purple', s=100, marker='o', label='Start/End Point')

    # Set labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    return optimized_routes

# Example usage
optimized_uav_paths = solve_uav_paths_with_ga(num_points=100, num_uavs=3)

# Print optimized paths for each UAV
for uav_id, route in optimized_uav_paths.items():
    print(f"\nOptimized Route for UAV {uav_id+1}:\n", route)
