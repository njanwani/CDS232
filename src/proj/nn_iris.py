import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sig(x)
        x = self.fc2(x)
        return x
    
class UltraSimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(UltraSimpleNN, self).__init__()
        self.layer1 = nn.Linear(3, output_size)
    
    def forward(self, x):
        # print(x.T[:3, :].T)
        return self.layer1(x.T[:3, :].T)

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
print('input size:', input_size)
hidden_size = 64
output_size = len(torch.unique(y_train_tensor))
model = UltraSimpleNN(input_size, output_size) #SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
batch_size = 2

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

    # Print training loss every 10 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = torch.sum(predicted == y_test_tensor).item() / len(y_test_tensor)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(X_test_tensor)
    print('---------')
    print(outputs)
    print('---------')
    
    
# Get the weight matrix of the first layer
weight_matrix = model.layer1.weight
matrix_to_save = weight_matrix.detach().numpy()
pickle_file_path = "iris_weights_ultrasimple_weights.pickle"

# Open the file in binary write mode and dump the matrix using pickle
with open(pickle_file_path, 'wb') as file:
    pickle.dump(matrix_to_save, file)
    
    
# Get the bias matrix of the first layer
bias_matrix = model.layer1.bias
matrix_to_save = bias_matrix.detach().numpy()
pickle_file_path = "iris_weights_ultrasimple_bias.pickle"

# Open the file in binary write mode and dump the matrix using pickle
with open(pickle_file_path, 'wb') as file:
    pickle.dump(matrix_to_save, file)
