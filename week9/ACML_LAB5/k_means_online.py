import numpy as np

np.random.seed(0)

dataset = np.random.randn(300, 2)

k = 5
centroids = dataset[np.random.choice(len(dataset), k, replace=False)]
learning_rate = 0.01


def min_euclidean_distance(point, centroids_param):
    distances = np.linalg.norm(np.array(point)-np.array(centroids_param), axis=1)
    min_index = np.argmin(distances)
    return min_index



counter = 0
while counter < 70:
    counter += 1
    cluster_means = {}
    cluster_dict = {i: [] for i in range(k)}
    for point in dataset:
        j = min_euclidean_distance(point, centroids)
        cluster_dict[j].append(point)
    for (key, value) in enumerate(cluster_dict):
        cluster_means[key] = np.mean(cluster_dict[key], axis=0)
    centroids = [cluster_means[key] for key in cluster_means]
    print(centroids)
