# (c) Jan Zwiener (jan@zwiener.org)
#
# Neural network architecture to imitate the Model Predictive
# Control outputs. Used by nnpolicy.py and expert_train.py
#

import torch
import torch.nn as nn

class NNPolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First MLP layer
        self.fc2 = nn.Linear(128, 256)         # Second MLP layer
        self.fc3 = nn.Linear(256, output_size) # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # x = torch.tanh(self.fc3(x))  # Apply tanh to the output of the final layer
        return x

