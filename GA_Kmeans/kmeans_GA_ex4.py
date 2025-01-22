import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Helper function to calculate Euclidean distance (Pythagoras)
def calculate_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

# Manual KMeans clustering using Pythagorean distance formula
def manual_kmeans_clustering(points, num_uavs, max_iter=300):
    # Step 1: Randomly initialize centroids
    np.random.seed(42)
    centroids = points[np.random.choice(points.shape[0], num_uavs, replace=False)]
    
    for iteration in range(max_iter):
        # Step 2: Assign points to the nearest centroid
        clusters = [[] for _ in range(num_uavs)]  # Empty lists for each cluster

        for point in points:
            distances = [calculate_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)

        # Step 3: Recompute centroids
        new_centroids = np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else centroids[i] for i, cluster in enumerate(clusters)])

        # Step 4: Check for convergence (if centroids do not change)
        if np.allclose(new_centroids, centroids):
            print(f"Converged after {iteration+1} iterations")
            break

        centroids = new_centroids  # Update centroids for next iteration

    # Print KMeans input and output
    print("\n--- KMeans Input ---")
    print("Points:")
    print(points)
    print(f"\nNumber of UAVs: {num_uavs}\n")
    print("--- KMeans Output ---")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1} Waypoints:")
        print(np.array(cluster))
    print("\nCentroids:")
    print(centroids)

    return clusters, centroids

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

    # Print GA Input and Output
    print("\n--- GA Input ---")
    print("Waypoints:")
    print(xyz)
    print("\n--- GA Output ---")
    print("Optimized Route:")
    print(rearranged_route)

    return rearranged_route

# Plotting the manually computed KMeans clusters
def plot_manual_kmeans(points, clusters, centroids, num_uavs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Manual KMeans Clustering with {num_uavs} UAVs')

    colors = cm.rainbow(np.linspace(0, 1, num_uavs))

    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=colors[i], label=f'UAV {i+1}')
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', s=100, marker='x', label='Centroids')

    # Plot start point
    ax.scatter(0, 0, 0, c='purple', s=100, marker='o', label='Start Point')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Solve paths for UAVs using KMeans + GA and plot results
def solve_uav_paths_with_kmeans_and_ga(num_points=100, num_uavs=3):
    # Step 1: Generate random 3D points
    np.random.seed(42)
    points = np.random.rand(num_points, 3) * 100

    # Step 2: Run manual KMeans clustering
    clusters, centroids = manual_kmeans_clustering(points, num_uavs)

    # Step 3: Plot KMeans results
    plot_manual_kmeans(points, clusters, centroids, num_uavs)

    # Step 4: Run GA for each UAV's set of points and plot results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Optimized UAV Paths with KMeans and GA for {num_uavs} UAVs')

    colors = cm.rainbow(np.linspace(0, 1, num_uavs))

    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        # Add the start point (0, 0, 0) to the cluster points for GA
        cluster_points = np.vstack([[0, 0, 0], cluster_points])

        # Run GA to optimize the path for the current UAV
        optimized_route = ga_3d_pathplanning(cluster_points)

        # Plot the optimized route
        ax.plot(optimized_route[:, 0], optimized_route[:, 1], optimized_route[:, 2], 'o-', color=colors[i], label=f'UAV {i+1}')

    # Plot start/end point
    ax.scatter(0, 0, 0, c='purple', s=100, marker='o', label='Start/End Point')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Example usage
solve_uav_paths_with_kmeans_and_ga(num_points=100, num_uavs=3)
