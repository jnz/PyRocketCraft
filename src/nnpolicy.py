from basepolicy import BasePolicy
from nnpolicynetwork import NNPolicyNetwork

import numpy as np
import torch
import torch.nn as nn

class NNPolicy(BasePolicy):
    def __init__(self, initial_state):
        super().__init__()

        self.model = torch.load('torch_nn_mpc-rocket-v0.pth').to('cpu')
        self.model.eval()  # Set the model to inference mode

    def predict(self, observation):

        state_tensor = torch.tensor(observation, dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():  # Disables gradient calculation, which is not needed during inference
            action_pred = self.model(state_tensor)
        action = action_pred.numpy()

        return action.ravel()

