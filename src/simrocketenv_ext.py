# (c) Jan Zwiener (jan@zwiener.org)

import numpy as np
from gymnasium import spaces

from simrocketenv import SimRocketEnv
from nnpolicy import NNPolicy

class SimRocketEnv_Ext(SimRocketEnv):
    def __init__(self, interactive=False):
        super().__init__(interactive)

        # self.policy = MPCPolicy(self.state)
        self.policy = NNPolicy()

        self.DELTAUMIN = -0.5
        self.DELTAUMAX =  0.5
        self.action_space = spaces.Box(low=np.float32(self.DELTAUMIN), high=np.float32(self.DELTAUMAX), shape=(self.ACTUATORCOUNT,), dtype=np.float32)

    def step(self, action):

        action0, predictedX = self.policy.next(self.state)
        return super().step(action + action0)

