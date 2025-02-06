import numpy as np
import matplotlib.pyplot as plt
import pickle

# GA Path Planning function
def ga_3d_pathplanning(xyz):
    N = xyz.shape[0]
    popSize = 8
    numIter = int(1e5)

    # Distance matrix
    a = np.meshgrid(np.arange(N), np.arange(N))
    dmat = np.sqrt(np.sum((xyz[a[0], :] - xyz[a[1], :])**2, axis=2))

    pop = np.zeros((popSize, N), dtype=int)
    pop[0, :] = np.arange(N)
    for k in range(1, popSize):
        pop[k, :] = np.random.permutation(N)

    globalMin = np.inf
    for iter in range(numIter):
        totalDist = np.zeros(popSize)
        for p in range(popSize):
            d = dmat[pop[p, -1], pop[p, 0]]
            for k in range(1, N):
                d += dmat[pop[p, k - 1], pop[p, k]]
            totalDist[p] = d

        minDist = np.min(totalDist)
        if minDist < globalMin:
            globalMin = minDist
            optRoute = pop[np.argmin(totalDist), :]

        newPop = np.zeros_like(pop)
        for p in range(0, popSize, 4):
            bestRoute = pop[p:p+4][np.argmin(totalDist[p:p+4])]
            newPop[p:p+4] = np.random.permutation(bestRoute)
        pop = newPop

    final_route = xyz[optRoute, :]
    return np.vstack((final_route, [xyz[0]]))

# Main function
if __name__ == "__main__":
    # Load KMeans output
    with open("kmeans_output.pkl", "rb") as f:
        clusters_with_start = pickle.load(f)

    # Solve GA for each UAV
    optimized_paths = []
    for i, cluster_points in enumerate(clusters_with_start):
        optimized_path = ga_3d_pathplanning(cluster_points)
        optimized_paths.append(optimized_path)

        # Plot individual UAV path
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(optimized_path[:, 0], optimized_path[:, 1], optimized_path[:, 2], label=f"UAV {i+1}")
        ax.scatter(0, 0, 0, c="red", s=100, marker="o", label="Start/End Point")
        ax.set_title(f"Optimized Path for UAV {i+1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

        # Print optimized path
        print(f"Optimized path for UAV {i+1}:")
        print(optimized_path)
        print()

    # Combined plot for all UAVs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.rainbow(np.linspace(0, 1, len(optimized_paths)))
    for i, path in enumerate(optimized_paths):
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=colors[i], label=f"UAV {i+1}")
    ax.scatter(0, 0, 0, c="red", s=100, marker="o", label="Start/End Point")
    ax.set_title("Combined Optimized Paths for All UAVs")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
