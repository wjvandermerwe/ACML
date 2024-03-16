from ucimlrepo import fetch_ucirepo
import ssl
import numpy as np
import torch

ssl._create_default_https_context = ssl._create_unverified_context

# 53 = iris, 545 = rice,  raisin = 850

data = fetch_ucirepo(id=545)
X = data.data.features
y = data.data.targets.values.flatten()

unique_vals = np.unique(y).tolist()
n_classes = len(unique_vals)
one_hot_matrix = np.zeros((len(y), n_classes))

for i, val in enumerate(y):
    index = unique_vals.index(val)
    one_hot_matrix[i, index] = 1

inputs = torch.tensor(X.values)
labels = torch.tensor(one_hot_matrix)

num_rows_inputs = inputs.shape[0]

q = num_rows_inputs // 4

inputs_indices = torch.arange(len(inputs))
shuffled_indices = torch.randperm(len(inputs_indices))

inputs = inputs[shuffled_indices]
labels = labels[shuffled_indices]

test_inputs = inputs[q*3:num_rows_inputs]
validation_inputs = inputs[q*2:q*3]
training_inputs = inputs[0:q*2]

test_labels = labels[q*3:num_rows_inputs]
validation_labels = labels[q*2:q*3]
training_labels = labels[0:q*2]

