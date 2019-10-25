import torch
import torch.nn as nn
import torch.nn.functional as F

class Feedforward(nn.Module):
    def __init__(self, in_size, out_size):
        super(Feedforward, self).__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, x):
        return F.softmax(self.fc(x))

class ValueNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, x):
        return self.fc(x)
