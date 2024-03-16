import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import time

def gen_data(N, behavior='periodic'):
    
    X = np.random.random((N, 2)) * 4 - 2
    # y = np.multiply(X[:,::-1], [-1, 1]) 
    
    if behavior == 'periodic':
        r = np.vstack((np.linalg.norm(X, axis=1), np.linalg.norm(X, axis=1)))
        thetas = np.arctan2(X[:, 1], X[:, 0]) + np.pi / 2 - 0.2 * np.tanh(0.1 * (2 - np.linalg.norm(X, axis=1)))
        y = 1 * np.vstack((np.cos(thetas),
                    np.sin(thetas))).T
    elif behavior == 'stable':
        r = np.vstack((np.linalg.norm(X, axis=1), np.linalg.norm(X, axis=1)))
        thetas = np.arctan2(X[:, 1], X[:, 0]) + np.pi + 0.5
        y = r.T * np.vstack((np.cos(thetas),
                    np.sin(thetas))).T
    elif behavior == 'unstable':
        r = np.vstack((np.linalg.norm(X, axis=1), np.linalg.norm(X, axis=1)))
        thetas = np.arctan2(X[:, 1], X[:, 0]) + 0.5
        y = r.T * np.vstack((np.cos(thetas),
                    np.sin(thetas))).T
    elif behavior == 'wave':
        thetas = np.sin(X[:, 0])
        y = np.vstack((np.cos(thetas),
                       np.sin(thetas))).T
    print(X.shape)
    print(y.shape)
    return X, y

BEHAVIOR = 'stable'
# Load and preprocess the Iris dataset
X, y = gen_data(1_000, behavior=BEHAVIOR)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sig = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sig(x)
        x = self.fc2(x)
        return x
    
class UltraSimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(UltraSimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size)
        self.nonlin = nn.Tanh()
    
    def forward(self, x):
        # print(x.T[:3, :].T)
        return 10 * self.nonlin(self.layer1(x)) - x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
print('input size:', input_size)
hidden_size = 64
output_size = y_train.shape[1]
# model = SimpleNN(input_size, hidden_size, output_size)
model = UltraSimpleNN(input_size, output_size) 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, verbose=False)
# Training loop
num_epochs = 51
epoch_saves = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
batch_size = 4

for epoch in range(num_epochs):
    model.train()

    # Mini-batch training
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        labels = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Print training loss every 10 epochs
    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
    if int(epoch) in epoch_saves:
        source = '/Users/neiljanwani/Documents/CDS232/src/project/learning/'
        # Get the weight matrix of the first layer
        weight_matrix = model.layer1.weight
        matrix_to_save = weight_matrix.detach().numpy()
        pickle_file_path = f"{BEHAVIOR}_A_alpha10_{epoch}.pickle"
        # Open the file in binary write mode and dump the matrix using pickle
        with open(source + pickle_file_path, 'wb') as file:
            pickle.dump(matrix_to_save, file)
            
            
        # Get the bias matrix of the first layer
        bias_matrix = model.layer1.bias
        matrix_to_save = bias_matrix.detach().numpy()
        pickle_file_path = f"{BEHAVIOR}_b_alpha10_{epoch}.pickle"

        # Open the file in binary write mode and dump the matrix using pickle
        with open(source + pickle_file_path, 'wb') as file:
            pickle.dump(matrix_to_save, file)
        
# Test the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    print(X_test_tensor.detach().numpy()[:10])
    print('---------')
    print(outputs.detach().numpy()[:10])
    print('---------')
    
    
# Get the weight matrix of the first layer
weight_matrix = model.layer1.weight
matrix_to_save = weight_matrix.detach().numpy()
pickle_file_path = f"{BEHAVIOR}_A_alpha10_{num_epochs}.pickle"

# Open the file in binary write mode and dump the matrix using pickle
with open(pickle_file_path, 'wb') as file:
    pickle.dump(matrix_to_save, file)
    
    
# Get the bias matrix of the first layer
bias_matrix = model.layer1.bias
matrix_to_save = bias_matrix.detach().numpy()
pickle_file_path = f"{BEHAVIOR}_b_alpha10_{num_epochs}.pickle"

# Open the file in binary write mode and dump the matrix using pickle
with open(pickle_file_path, 'wb') as file:
    pickle.dump(matrix_to_save, file)
