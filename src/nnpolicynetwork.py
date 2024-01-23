import torch
import torch.nn as nn

class NNPolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First MLP layer
        self.fc2 = nn.Linear(128, 128)         # Second MLP layer
        self.fc3 = nn.Linear(128, output_size) # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

