#!/usr/bin/env python3

import numpy as np
import scipy.linalg
import time
from simrocketenv import SimRocketEnv
from geodetic_toolbox import *
import threading
import copy
import json # log action <-> obs pairs

from mpcpolicy import MPCPolicy
from nnpolicy import NNPolicy
from ppopolicy import PPOPolicy

# Global messagebox to exchange data between threads as shown above
g_thread_msgbox = {
    'keymap' : { 'longitudinal_cmd': 0.0, 'lateral_cmd': 0.0, 'yaw_cmd': 0.0, 'vertical_cmd': 0.0, 'yaw_cmd': 0.0, }, # keyboard input
    'mpc_fps' : 0,                     # debug information: fps of NMPC thread
    'render_fps' : 0,                  # debug information: fps of render thread
}
g_thread_msgbox_lock = threading.Lock() # only access g_thread_msgbox with this lock
g_sim_running = True # Run application as long as this is set to True

# This thread's job is to consume the simulation state vector and emit a
# control output u (that is again consumed by the physics simulation)
def nmpc_thread_func(initial_state):
    global g_thread_msgbox
    global g_thread_msgbox_lock
    global g_sim_running

    # policy = MPCPolicy(initial_state)
    policy = NNPolicy()
    # policy = PPOPolicy()
    print("Active policy: %s" % (policy.get_name()))

    # Add-on:
    # Keep track of observation and action vectors of the MPC to pre-train a Neural Network
    # Set collect_training_data to True to collect training data in a json file
    # Call expert_train.py later to process data
    collect_training_data = False
    expert_data = []
    if collect_training_data:
        try:
            with open("expert_data.json", "r") as f:
                expert_data = json.load(f)
        except Exception as e:
            print(e)

    # # make sure a MPC update is performed in the first epoch
    MPC_DT_SEC = 1.0 / 100.0  # run the NMPC every XX ms
    timestamp_last_mpc_update = time.time() - 2*MPC_DT_SEC
    mpc_step_counter = 0  # +1 for every mpc step, reset every 1 sec
    timestamp_last_mpc_fps_update = time.time()

    while g_sim_running:

        timestamp_current = time.time()
        if timestamp_current - timestamp_last_mpc_update < MPC_DT_SEC:
            time.sleep(0)
            continue

        with g_thread_msgbox_lock:
            keymap = copy.deepcopy(g_thread_msgbox['keymap']) # read input from render thread
            state = copy.deepcopy(g_thread_msgbox['state'])
            if timestamp_current - timestamp_last_mpc_fps_update >= 1.0:
                g_thread_msgbox['mpc_fps'] = mpc_step_counter
                mpc_step_counter = 0
                timestamp_last_mpc_fps_update = timestamp_current

        u, predictedX = policy.next(state)

        expert_data.append({"obs": state.tolist(), "acts": u.tolist(), "predictedX": predictedX.tolist() })

        timestamp_last_mpc_update = timestamp_current
        mpc_step_counter += 1
        with g_thread_msgbox_lock:
            g_thread_msgbox['u'] = np.copy(u) # output control vector u

    if collect_training_data and policy.get_name() == "MPC":
        print("Dumping data to .json file...")
        with open("expert_data.json", "w") as f:
            json.dump(expert_data, f, indent=4, sort_keys=True)
        print("done")

def main():
    global g_thread_msgbox
    global g_thread_msgbox_lock
    global g_sim_running


    # SimRocketEnv is handling the physics simulation
    env = SimRocketEnv(interactive=True)
    g_thread_msgbox['state'] = env.state
    u = None
    predictedX = None

    # Spawn NMPC thread (doing the control work)
    nmpc_thread = threading.Thread(target=nmpc_thread_func, kwargs={'initial_state': g_thread_msgbox['state']})
    nmpc_thread.start()

    timestamp_lastupdate = time.time()
    MAX_DT_SEC = 0.1 # don't allow larger simulation timesteps than this
    SIM_DT_SEC = 1.0 / 120.0  # run the simulation every XX ms
    sim_step_counter = 0  # +1 for every simulation step, reset every 1 sec
    # emit a FPS stat message every second based on this timestamp:
    last_fps_update = timestamp_lastupdate
    reward_sum = 0.0

    while g_sim_running:
        timestamp_current = time.time()
        dt_sec = timestamp_current - timestamp_lastupdate
        if dt_sec < SIM_DT_SEC:
            time.sleep(0)
            continue

        with g_thread_msgbox_lock:
            if 'u' in g_thread_msgbox:
                u = copy.deepcopy(g_thread_msgbox['u']) # get control vector u from NMPC thread
            else:
                continue
        # wait with the simulation thread until we receive the first u vector
        # e.g. the MPC thread needs to compile .c files until it is ready

        timestamp_lastupdate = timestamp_current
        # make sure dt_sec is within a reasonable range
        if dt_sec > MAX_DT_SEC:
            print("Warning, high dt_sec: %.1f" % (dt_sec))
            continue
        dt_sec = np.clip(dt_sec, 0.0, MAX_DT_SEC)

        env.dt_sec = dt_sec
        state, reward, done, tr, info = env.step(u) # update physics simulation
        sim_step_counter += 1
        reward_sum += reward

        if done == True:
            g_sim_running = False

        with g_thread_msgbox_lock:
            g_thread_msgbox['state'] = state
            mpc_fps       = g_thread_msgbox['mpc_fps']
            render_fps    = g_thread_msgbox['render_fps']

        if timestamp_current - last_fps_update >= 0.1:
            print("FPS=%3i SIM=%4i MPC=%3i score=%i" % (render_fps, sim_step_counter, mpc_fps, reward_sum), end=' ')
            last_fps_update = timestamp_current

            sim_step_counter = 0

            env.print_state()
            for elem in u:
                print("%3.0f" % (elem*99.0), end=' ')
            print("")

    nmpc_thread.join()

if __name__ == '__main__':
    main()

