# (c) Jan Zwiener (jan@zwiener.org)
#
# Experimental code
#
# Use PPO to extend the existing NNPolicy by
# delta actions to improve the behavior.
# This will generate a new model saved in "ppo-rocket-vX".
# This model file is used by the PPOPolicy which
# is running internally the NNPolicy + the delta
# on the actions trained here.

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from gymnasium import spaces
from simrocketenv import SimRocketEnv
from nnpolicy import NNPolicy

# Wrapper for SimRocketEnv so that by default NNPolicy
# control inputs are added to the simulation.
# This environment extends simrocketenv.py
# so that the default neural network control input is
# always added to the input to this environment.
# In this way e.g. the PPO training can add delta
# corrections to the existing controller.
class SimRocketEnv_Ext(SimRocketEnv):
    def __init__(self, interactive=False):
        super().__init__(interactive)

        self.policy = NNPolicy()

        self.DELTAUMIN = -0.5
        self.DELTAUMAX =  0.5
        self.action_space = spaces.Box(low=np.float32(self.DELTAUMIN),
                                       high=np.float32(self.DELTAUMAX),
                                       shape=(self.ACTUATORCOUNT,),
                                       dtype=np.float32)

    def step(self, action):

        action0, predictedX = self.policy.next(self.state)
        return super().step(action + action0)

# Helper function for parallel learning
def make_env():
    def _init():
        env = SimRocketEnv_Ext(interactive=False)
        return env
    return _init

def train_and_evaluate():
    num_envs = 1 # run X environments in parallel
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        learning_rate=3e-5,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.04,
        vf_coef=0.5,
        max_grad_norm=1.0,
        device="cuda"
    )

    model.learn(total_timesteps=90000)
    model_name = 'ppo-rocket-v1'
    model.save(model_name)

if __name__ == '__main__':
    train_and_evaluate()
