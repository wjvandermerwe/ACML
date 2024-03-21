from ucimlrepo import fetch_ucirepo, list_available_datasets
import ssl
import numpy as np
import torch

# Ignore ssl certificate verification
ssl._create_default_https_context = ssl._create_unverified_context
iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets.values.flatten()
# print(y)

inputs = torch.tensor(X.values)
print(inputs)

unique_vals = np.unique(y).tolist()
n_classes = len(unique_vals)
one_hot_matrix = np.zeros((len(y), n_classes))

for i, val in enumerate(y):
    index = unique_vals.index(val)
    one_hot_matrix[i, index] = 1
labels= torch.tensor(one_hot_matrix)
# print(labels)

# print(one_hot_matrix)

