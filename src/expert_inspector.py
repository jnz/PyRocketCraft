"""
  Compare a bunch random data points between the MPC
  and the NN for debugging.

  (c) Jan Zwiener (jan@zwiener.org)
"""

from simrocketenv import SimRocketEnv
from mpcpolicy import MPCPolicy
from nnpolicy import NNPolicy

def print_vector(u, fstr):
    """
      Print a vector with a format string (fstr)
    """
    print("[", end=" ")
    for i in range(len(u)):
        print(fstr % (u[i]), end=" ")
    print("]")

def expert_collect():
    """
      Generate a random state vector and see how the MPC reacts
      and how the NN reacts.
    """

    # Settings:
    MAX_EPISODES = 10 # how many trajectories should be generated?
    STATE_SPACE_SCALE = 0.1

    # Generate simulation and controller object:
    env = SimRocketEnv(interactive=False, scale_obs_space=STATE_SPACE_SCALE)
    mpcpolicy = MPCPolicy(env.state)
    nnpolicy = NNPolicy()
    rng_seed = 0

    # training loop
    for episode in range(MAX_EPISODES):
        done = False
        state, _ = env.reset(seed=rng_seed) # start new trajectory
        rng_seed += 1

        u_mpc, _ = mpcpolicy.next(state)
        u_nn, _ = nnpolicy.next(state)

        env.print_state()
        print("MPC: ", end="")
        print_vector(u_mpc, "%6.3f")
        print("NN:  ", end="")
        print_vector(u_nn, "%6.3f")


if __name__ == '__main__':
    expert_collect()
