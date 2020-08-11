import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_shape, out_shape, dropout_p=0.01):
        super(DQN, self).__init__()

        self.input_dims = input_shape.shape[0]
        self.hidden_dims = 200
        self.output_dims = out_shape.n

        self.fc1 = nn.Linear(self.input_dims, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc3 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc4 = nn.Linear(self.hidden_dims, self.output_dims)
        self.rl = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims)
        x = self.dropout(self.rl(self.fc1(x)))
        x = self.rl(self.fc2(x))
        x = self.rl(self.fc3(x))
        return self.fc4(x.view(x.size(0), -1))
