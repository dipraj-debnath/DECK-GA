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
    print(f"Number of points (N): {N}")
    print(f"Population size (popSize): {popSize}")
    print(f"Number of iterations (numIter): {numIter}")

    # Generate distance matrix in 3D
    a = np.meshgrid(np.arange(N), np.arange(N))
    dmat = np.round(np.sqrt(np.sum((xyz[a[0], :] - xyz[a[1], :])**2, axis=2)))
    print("Distance matrix (dmat):\n", dmat)

    # Sanity checks for population size
    popSize = 4 * int(np.ceil(popSize / 4))
    numIter = max(1, round(numIter))
    print(f"Adjusted population size: {popSize}")
    print(f"Adjusted number of iterations: {numIter}")

    # Initialize population
    pop = np.zeros((popSize, N), dtype=int)
    pop[0, :] = np.arange(N)  # Start with sequential route
    for k in range(1, popSize):
        pop[k, :] = np.random.permutation(N)  # Random routes for the population
    print("Initial population:\n", pop)

    # GA variables
    globalMin = np.inf
    distHistory = np.zeros(numIter)
    
    # Figure for plotting
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("TSP_GA | 3D Current Best Solution")

    # Main GA loop
    for iter in range(numIter):
        totalDist = np.zeros(popSize)
        
        # Calculate total distance for each member of the population
        for p in range(popSize):
            d = dmat[pop[p, -1], pop[p, 0]]  # Closed path
            for k in range(1, N):
                d += dmat[pop[p, k - 1], pop[p, k]]
            totalDist[p] = d
        #print(f"Iteration {iter+1}: Total distances:\n", totalDist)

        # Find the best route
        minDist = np.min(totalDist)
        index = np.argmin(totalDist)
        distHistory[iter] = minDist
        #print(f"Iteration {iter+1}: Best distance: {minDist}")

        if minDist < globalMin:
            globalMin = minDist
            optRoute = pop[index, :]

            # Plot the best route in 3D
            ax.clear()
            rte = np.append(optRoute, optRoute[0])  # Complete the cycle
            ax.plot(xyz[rte, 0], xyz[rte, 1], xyz[rte, 2], 'r.-', linewidth=2)
            ax.set_title(f'Total Distance = {minDist:.4f}, Iteration = {iter+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.draw()
            plt.pause(0.01)

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
                tmpPop[1, :] = bestOf4Route  # If I == J, do nothing (no flip)

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
    
    plt.ioff()

    # Final output and plots
    fig_final = plt.figure()
    ax1 = fig_final.add_subplot(211, projection='3d')
    rte = np.append(optRoute, optRoute[0])
    ax1.plot(xyz[rte, 0], xyz[rte, 1], xyz[rte, 2], 'r.-', linewidth=2)
    ax1.set_title(f'Best Route: Total Distance = {globalMin:.4f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig_final.add_subplot(212)
    ax2.plot(distHistory, 'b', linewidth=2)
    ax2.set_title('Distance History')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Distance')
    ax2.grid(True)
    
    plt.show()

    # Post-processing: Rearrange final route to start from [0, 0, 0]
    final_route = xyz[optRoute, :]
    print("Final optimized route before reordering:\n", final_route)

    start_point = np.where(np.all(final_route == xyz[0], axis=1))[0][0]
    rearranged_route = np.concatenate((final_route[start_point:], final_route[:start_point+1]))

    print("Rearranged final optimized route:\n", rearranged_route)

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

    # Step 4: Plotting the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Dynamically create colors for the UAVs based on the number of UAVs
    colors = cm.rainbow(np.linspace(0, 1, num_uavs))

    # Store points for each UAV cluster
    uav_clusters = {}
    for i in range(num_uavs):
        cluster_points = points[labels == i]
        uav_clusters[i] = cluster_points  # Save the points for each UAV

        # Plot points with different colors for each cluster
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=[colors[i]], label=f'UAV {i+1}')

    # Plot centroids
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', s=100, marker='x', label='Centroids')

    # Starting point for all UAVs
    ax.scatter(0, 0, 0, c='purple', s=100, marker='o', label='Start Point')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    return uav_clusters  # Return the clusters to use in GA

# Solve paths for UAVs using GA
def solve_uav_paths_with_ga(num_points=100, num_uavs=3):
    # Step 1: Run K-Means clustering to divide points among UAVs
    uav_clusters = kmeans_clustering_3d(num_points=num_points, num_uavs=num_uavs)

    # Step 2: Run GA for each UAV's set of points
    optimized_routes = {}
    for uav_id, uav_points in uav_clusters.items():
        print(f"\nOptimizing path for UAV {uav_id+1} with {uav_points.shape[0]} points.")
        optimized_routes[uav_id] = ga_3d_pathplanning(uav_points)

    return optimized_routes

# Example usage
optimized_uav_paths = solve_uav_paths_with_ga(num_points=100, num_uavs=3)
