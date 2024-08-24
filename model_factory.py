import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def build_model(name, row, col, input_size, output_size):
    if name == "NN1":
        return create_NN1(input_size, output_size)
    elif name == "NN2":
        return create_NN2(input_size, output_size)
    elif name == "CNN":
        return create_CNN(row, col, output_size)
    elif name == "RNN":
        return create_RNN(input_size, output_size)
    elif name == "Transformer":
        return create_Transformer(input_size, output_size)
    else:
        raise ValueError(f"Unknown model name: {name}")


def create_NN1(input_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, int(input_size//2)),
        nn.ReLU(),
        nn.Linear(int(input_size//2), output_size)
    )
    optimizer = optim.RAdam(model.parameters())
    return model, optimizer


def create_NN2(input_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, input_size),
        nn.ReLU(),
        nn.Linear(input_size, int(input_size//2)),
        nn.ReLU(),
        nn.Linear(int(input_size//2), output_size)
    )
    optimizer = optim.Adam(model.parameters())
    return model, optimizer


def create_CNN(row, col, output_size):
    class CNNModel(nn.Module):
        def __init__(self, in_channels, rows, cols, output_size):
            super(CNNModel, self).__init__()
            self.c1_kernel = 3
            self.c1_out = 8
            self.conv1 = nn.Conv2d(
                in_channels=in_channels, out_channels=self.c1_out, kernel_size=self.c1_kernel)
            self.pool = nn.MaxPool2d(kernel_size=2)

            self.rows = rows
            self.cols = cols
            self._input_fc1 = self.calc_size_fc1()

            self.fc1 = nn.Linear(self._input_fc1, 16)
            self.fc2 = nn.Linear(16, output_size)

        def calc_size_fc1(self):
            # after conv1 and pooling
            rows = (self.rows - (self.c1_kernel-1)) // 2
            cols = (self.cols - (self.c1_kernel-1)) // 2

            return rows * cols * self.c1_out

        def forward(self, x):
            x = x.view(-1, 1, self.rows, self.cols)

            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(-1, self._input_fc1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = CNNModel(1, row, col, output_size)
    optimizer = optim.Adam(model.parameters())
    return model, optimizer


def create_RNN(input_size, output_size, hidden_size=64):
    class RNNModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNNModel, self).__init__()

            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            _, hidden = self.rnn(x)
            output = self.fc(hidden.squeeze(0))
            return output

    model = RNNModel(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters())
    return model, optimizer


def create_Transformer(input_size, output_size, nhead=4, num_layers=2):
    class TransformerModel(nn.Module):
        def __init__(self, input_size, output_size, nhead, num_layers):
            super(TransformerModel, self).__init__()

            self.transformer = nn.Transformer(
                d_model=input_size,
                nhead=nhead,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                dim_feedforward=128,
                batch_first=True
            )
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, src, tgt):
            output = self.transformer(src, tgt)
            return self.fc(output)

    model = TransformerModel(input_size, output_size, nhead, num_layers)
    optimizer = optim.Adam(model.parameters())
    return model, optimizer
