import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Build the NN model based on its type and hyperparametrs
def build_model(name, row, col, input_size, output_size):
    if name == "NN1":
        return create_NN1(input_size, output_size)
    elif name == "CNN":
        return create_CNN(row, col, output_size)
    else:
        raise ValueError(f"Unknown model name: {name}")

# Create a one hidden fully connected layer NN
def create_NN1(input_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, int(input_size)),
        nn.ReLU(),
        nn.Linear(int(input_size), output_size)
    )
    optimizer = optim.RAdam(model.parameters())
    return model, optimizer

# Create a CNN (small size states)
def create_CNN(row, col, output_size):
    class CNNModel(nn.Module):
        def __init__(self, in_channels, rows, cols, output_size):
            super(CNNModel, self).__init__()
            self.c1_kernel = 3
            self.c1_out = 8
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.c1_out, kernel_size=self.c1_kernel)
            self.pool = nn.MaxPool2d(kernel_size=2)

            self.rows = rows
            self.cols = cols
            self._input_fc1 = self.calc_size_fc1()

            self.fc1 = nn.Linear(self._input_fc1, 16)
            self.fc2 = nn.Linear(16, output_size)

        # Calculate the new input size for the fully connected layer
        def calc_size_fc1(self):
            rows = (self.rows - (self.c1_kernel-1)) // 2 # After conv1 and pooling
            cols = (self.cols - (self.c1_kernel-1)) // 2
            return rows * cols * self.c1_out

        def forward(self, x):
            is_multiple_states = False
            if x.dim() == 1:
                x = x.view(1, 1, self.rows, self.cols)
            elif x.dim() == 2:
                x = x.view(-1, 1, self.rows, self.cols)
                is_multiple_states = True 
            elif x.dim() == 3:
                x = x.unsqueeze(1)

            x = F.relu(self.conv1(x))
            x = self.pool(x)
            
            if is_multiple_states: 
                x = x.view(-1, self._input_fc1)
            else:
                x = x.view(self._input_fc1)

            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = CNNModel(1, row, col, output_size)
    optimizer = optim.Adam(model.parameters())
    return model, optimizer
