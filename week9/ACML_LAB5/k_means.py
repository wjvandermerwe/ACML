import math

dataset = [ [0.22, 0.33],
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
            [0.44, 0.55] ] 

k = 3


centroids = [(float(input()), float(input())) for _ in range(3)]
previous_centroids = None

def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))) # nice little hack

def compute_tolerance():
    # any of the previous centroid distances still moving? order matters
    return any(euclidean_distance(centroids[i], previous_centroids[i]) > 1e-10 for i in range(len(centroids)))

counter = 0
while (previous_centroids is None or compute_tolerance() and counter < 200):
    counter += 1
    previous_centroids = centroids[:]
    cluster_dict = {i: [] for i in range(len(centroids))}

    for point in dataset:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        min_dist = distances.index(min(distances))

        cluster_dict[min_dist].append(point)

    for i in range(len(centroids)):
        if cluster_dict[i]:
            centroids[i] = (sum(x for x, _ in cluster_dict[i]) / len(cluster_dict[i]), sum(y for _, y in cluster_dict[i]) / len(cluster_dict[i]))



def sum_of_squares_loss():
    return sum(min(euclidean_distance(point, centroid) ** 2 for centroid in centroids) for point in dataset)



output = sum_of_squares_loss()
    
print(round(output,4))  
