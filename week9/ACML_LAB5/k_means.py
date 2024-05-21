import numpy as np
import random

dataset = np.array([[0.22, 0.33],
                    [0.45, 0.76],
                    [0.73, 0.39],
                    [0.25, 0.35],
                    [0.51, 0.69],
                    [0.69, 0.42],
                    [0.41, 0.49],
                    [0.15, 0.29],
                    [0.81, 0.32],
                    [0.50, 0.88],
                    [0.23, 0.31],
                    [0.77, 0.30],
                    [0.56, 0.75],
                    [0.11, 0.38],
                    [0.81, 0.33],
                    [0.59, 0.77],
                    [0.10, 0.89],
                    [0.55, 0.09],
                    [0.75, 0.35],
                    [0.44, 0.55]])

k = 3
centroids = np.array([(float(input()), float(input())) for _ in range(k)])
previous_centroids = None

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def compute_tolerance():
    return not np.allclose(centroids, previous_centroids, atol=1e-30)


counter = 0
while previous_centroids is None or (compute_tolerance() and counter < 200):
    counter += 1
    previous_centroids = centroids.copy()
    cluster_dict = {i: [] for i in range(len(centroids))}

    for point in dataset:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        min_dist = distances.index(np.min(distances))
        cluster_dict[min_dist].append(point)

    new_centroids = []
    keys_to_remove = []

    for i in range(len(centroids)):
        if len(cluster_dict[i]) > 0:  # Check if the cluster is not empty
            new_centroids.append(np.mean(cluster_dict[i], axis=0))
    
    centroids = np.array(new_centroids)

    centroids = np.array([np.mean(cluster_dict[i], axis=0) for i in range(len(centroids)) if len(cluster_dict[i]) > 0])

def sum_of_squares_loss():
    return np.sum([min(np.square(euclidean_distance(point, centroid)) for centroid in centroids) for point in dataset])


output = sum_of_squares_loss()
print(round(output, 4))
