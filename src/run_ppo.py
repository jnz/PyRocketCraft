import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from simrocketenv import SimRocketEnv

# Load the PPO model from file
model_name = 'ppo-rocket-v0'
model = PPO.load(model_name)

# Create environment
eval_env = SimRocketEnv(interactive=True)

# Run the model in the environment
# obs = eval_env.reset()
obs = eval_env.state
cumulative_reward = 0

SIM_DT_SEC = 1.0/60.0
timestamp_lastupdate = time.time() - 2*SIM_DT_SEC

dones = False
while dones == False:
    timestamp_current = time.time()
    dt_sec = timestamp_current - timestamp_lastupdate
    dt_sec = np.clip(dt_sec, 0.0, 0.05)
    eval_env.dt_sec = dt_sec

    if dt_sec < SIM_DT_SEC:
        time.sleep(0)
        continue

    timestamp_lastupdate = time.time()
    action, _states = model.predict(obs, deterministic=True)

    obs, rewards, dones, info = eval_env.step(action)
    eval_env.render()
    cumulative_reward += rewards

print(f"Cumulative reward: {cumulative_reward}")
print("Epochs: %i Time: %.3f s Reset Count: %i" % (eval_env.epochs, eval_env.time_sec, eval_env.reset_count))

# wait until user presses enter then close the window
input("Press Enter to close the window")

