import numpy as np

# GA Path Planning function
def ga_3d_pathplanning(xyz):
    N = xyz.shape[0]  # Number of points
    popSize = 8       # Population size
    numIter = int(1e5)  # Number of iterations

    # Generate distance matrix in 3D
    a = np.meshgrid(np.arange(N), np.arange(N))
    dmat = np.round(np.sqrt(np.sum((xyz[a[0], :] - xyz[a[1], :])**2, axis=2)))

    # Sanity checks for population size
    popSize = 4 * int(np.ceil(popSize / 4))
    pop = np.zeros((popSize, N), dtype=int)
    pop[0, :] = np.arange(N)
    for k in range(1, popSize):
        pop[k, :] = np.random.permutation(N)

    # GA variables
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
        randomOrder = np.random.permutation(popSize)
        newPop = np.zeros((popSize, N), dtype=int)
        for p in range(0, popSize, 4):
            rtes = pop[randomOrder[p:p+4], :]
            idx = np.argmin(totalDist[randomOrder[p:p+4]])
            bestOf4Route = rtes[idx, :]
            I, J = np.sort(np.random.randint(0, N, size=2))
            tmpPop = np.zeros((4, N), dtype=int)
            tmpPop[0, :] = bestOf4Route
            tmpPop[1, I:J+1] = bestOf4Route[I:J+1][::-1]
            tmpPop[2, :] = bestOf4Route
            tmpPop[2, [I, J]] = bestOf4Route[[J, I]]
            tmpPop[3, :] = bestOf4Route
            tmpPop[3, I:J+1] = np.roll(bestOf4Route[I:J+1], shift=-1)
            newPop[p:p+4, :] = tmpPop
        pop = newPop
    final_route = xyz[optRoute, :]
    start_point = np.where(np.all(final_route == xyz[0], axis=1))[0][0]
    rearranged_route = np.concatenate((final_route[start_point:], final_route[:start_point+1]))
    return rearranged_route
