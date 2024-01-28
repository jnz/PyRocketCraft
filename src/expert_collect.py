"""
  Collect training data from the "expert" controller and
  save it to a file (expert_data.json).

  (c) Jan Zwiener (jan@zwiener.org)
"""

import time
import json # log action <-> obs pairs

from simrocketenv import SimRocketEnv
from mpcpolicy import MPCPolicy

def expert_collect():
    """
      Run the MPC policy to generate sample trajectories and save
      the results to a file (defined by OUTPUT_FILE variable).
    """

    # Settings:
    OUTPUT_FILE = "expert_data.json" # where to write the results
    MAX_EPISODES = 400 # how many trajectories should be generated?
    ADD_PREVIOUS_RESULTS = True # overwrite previous training data?
    TIME_HORIZON = 15.0 # MPC prediction horizon
    EPOCHS_PER_SECOND = 20 # MPC epochs per second
    STATE_SPACE_SCALE = 0.8

    # Generate simulation and controller object:
    env = SimRocketEnv(interactive=True, scale_obs_space=STATE_SPACE_SCALE)
    policy = MPCPolicy(env.state,
                       time_horizon=TIME_HORIZON,
                       epochs_per_sec=EPOCHS_PER_SECOND)
    print("Active expert policy: %s" % (policy.get_name()))

    expert_data = []
    # load previous results and include it in the output?
    if ADD_PREVIOUS_RESULTS:
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                expert_data = json.load(f)
        except Exception as e:
            print("Failed to load previous results:", e)

    epoch_fps_counter = 0 # show FPS counter during collection
    total_epochs = 0 # keep track of epochs rendered so far
    last_fps_update = 0 # timestamp of last status update on stdout
    last_reward_sum = 0

    # training loop
    for episode in range(MAX_EPISODES):
        done = False
        reward_sum = 0
        state, _ = env.reset() # start new trajectory

        while not done:
            # Generate control input:
            u, predictedX = policy.next(state)

            # Simulation step:
            state, reward, done, _, _ = env.step(u) # update physics simulation
            reward_sum += reward
            if done:
                last_reward_sum = reward_sum

            # Save results
            expert_data.append({ "obs": state.tolist(),
                                 "acts": u.tolist(),
                                 "predictedX": predictedX.tolist() })

            timestamp_current = time.time()
            epoch_fps_counter += 1
            total_epochs += 1

            # Print some stats once per second:
            if timestamp_current - last_fps_update >= 1.0:
                progress = 100.0 * (episode / MAX_EPISODES)
                print("%3.0f%%. Total Epochs %i FPS: %i Last reward: %i" % (progress,
                                                    total_epochs,
                                                    epoch_fps_counter,
                                                    last_reward_sum))
                last_fps_update = timestamp_current
                epoch_fps_counter = 0

    print("Dumping data to .json file: %s" % (OUTPUT_FILE))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(expert_data, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    expert_collect()
