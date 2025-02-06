# DECK-GA A Hybrid KMeans Clustering and Genetic Algorithm for Optimised Mulit-UAV Path Planning in Multi-Traveling Salesman Problems
This repository presents DECK-GA, a hybrid algorithmic framework that combines DynamicCentroid Kmeans (DCKmeans) clustering with Distance Efficient Genetic Algorithm (DEGA) to solve path planning problems such as centroid instability, 
suboptimal UAV path distribution. This approach improves centroid initialisation and integration in DCKmeans to produce stable cluster formations. 
DEGA also improves trajectories  through a fitness-proportionate selection mechanism and an adaptive crossover mutation method, effectively increasing solution diversity and accelerating convergence.
# Usage
This section describes how to run the various Python scripts included in the DECK-GA framework to perform multi-UAV path planning. 
Below is a examples of description of DECK_GA script and instructions on how to execute them:

# DECK-GA: DynamicCentroid KMeans and Genetic Algorithm Framework

## Prerequisites

Before running the scripts, ensure you have Python installed on your system along with the following packages: numpy, matplotlib, scikit-learn, and pickle. Install these packages using pip:

```bash
pip install numpy matplotlib scikit-learn pickle5


Scripts and Their Execution Order:

1. Generate Random Points:
Generate_30_points.py - Generates 30 random 3D waypoints 
Generate_100_points.py - Generates 100 random 3D waypoints.

Command to run:
python Generate_30_points.py
python Generate_100_points.py

2. Clustering Algorithms:
Kmeans_2.py - Implements the classical KMeans algorithm.
kmeans_Updated_3.py - Implements the DynamicCentroid KMeans (DCKmeans) algorithm

Command to run:
python Kmeans_2.py
python kmeans_Updated_3.py

3. Path Planning with Genetic Algorithm
GA_path_planning.py - Executes the Distance Efficient Genetic Algorithm (DEGA) for path planning.

Command to run:
python GA_path_planning.py

4. Integrated KMeans and Genetic Algorithm
kmeans_GA_5.py - Combines KMeans clustering and GA to run the path planning process.

Command to run:
python kmeans_GA_5.py

5. Repeated Execution for Statistical Analysis
run_kmeans_GA_10_times.py - This script runs `kmeans_GA_5.py` ten times to generate and save results for further analysis. 

Command to run:
python run_kmeans_GA_10_times.py


```
sudo apt-get install python3-rpi.gpio
sudo pip3 install gpiozero
```
# Examples of generated paths
Put figures here
![servo connection](https://github.com/fervanegas/Test_360servo/blob/main/img/servo_connection2.jpg)
