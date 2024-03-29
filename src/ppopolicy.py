# (c) Jan Zwiener (jan@zwiener.org)
#
# Experimental code
#
# This policy needs a model that is generated by train_ppo.py
# The PPO policy is basically running NNPolicy and
# is using a delta action vector to correct/improve the
# action by NNPolicy.

import numpy as np
from stable_baselines3 import PPO

from basecontrol import BaseControl
from nnpolicy import NNPolicy

class PPOPolicy(BaseControl):
    def __init__(self):
        super().__init__()

        self.policy = NNPolicy() # Basic behavior is coming from NNPolicy
        model_name = 'ppo-rocket-v0' # Load the PPO model from file
        self.model = PPO.load(model_name)

    def get_name(self):
        return "PPO"

    def next(self, observation):
        # Calculate the basic NNPolicy action
        action0, predictedX = self.policy.next(observation)
        # Calculate corrective action vector
        action, _states = self.model.predict(observation, deterministic=True)
        # Return the sum of both models
        return action+action0, predictedX


