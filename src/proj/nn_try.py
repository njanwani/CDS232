import torch
import numpy as np
model = torch.nn.Linear(10, 10)

# Get the weight matrix of the first layer
weight_matrix = model.weight
bias_matrix = model.bias

# Print the weight matrix
print(weight_matrix.detach().numpy())
print(bias_matrix.detach().numpy())