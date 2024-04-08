import numpy as np

# Define the data
data = np.array([
    [0.8, 2.7],
    [0.7, 2.3],
    [0.2, 1.7],
    [0.3, 2.1],
    [0.3, 2.0],
    [0.1, 1.3],
    [0.4, 2.9],
    [0.1, 1.4]
])

split_points = [0.25,0.50,0.75]

for point in split_points:
    # Split the data based on the split point
    left_split = data[data[:, 0] <= point]
    right_split = data[data[:, 0] > point]

    # Calculate the mean squared error (MSE) for each group
    mse_left = np.mean((left_split[:, 1] - np.mean(left_split[:, 1]))**2)
    mse_right = np.mean((right_split[:, 1] - np.mean(right_split[:, 1]))**2)

    # Calculate the weighted average of the MSEs
    weighted_mse = (len(left_split) * mse_left + len(right_split) * mse_right) / len(data)

    # Calculate the MSE for the entire dataset 
    parent_mse = np.mean((data[:, 1] - np.mean(data[:, 1]))**2)

    # Calculate the gain from this split
    gain = parent_mse - weighted_mse

    print(f"split point {point}: ", gain)
