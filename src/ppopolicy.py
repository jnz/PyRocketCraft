import numpy as np
from stable_baselines3 import PPO

from basecontrol import BaseControl
from nnpolicy import NNPolicy

class PPOPolicy(BaseControl):
    def __init__(self):
        super().__init__()

        self.policy = NNPolicy()
        # Load the PPO model from file
        model_name = 'ppo-rocket-v0'
        self.model = PPO.load(model_name)

    def get_name(self):
        return "PPO"

    def next(self, observation):

        action0, predictedX = self.policy.next(observation)
        action, _states = self.model.predict(observation, deterministic=True)

        return action+action0, predictedX


