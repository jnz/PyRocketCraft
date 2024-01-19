import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from imitation.algorithms import bc
from imitation.data import types, rollout

from simrocketenv import SimRocketEnv

def train_and_evaluate():

    env = SimRocketEnv(interactive=False)

    expert_data = []
    obs = env.reset()
    done = False

    while not done:
        # action = mpc_controller.get_action(obs)
        expert_data.append({"obs": obs, "acts": action})
        obs, reward, done, info = env.step(action)

    trajectories = []
    current_trajectory = []

    # for each run:
    for obs, act in expert_data:
        current_trajectory.append(types.Transition(obs, act, None, None))  # None for reward and next_obs, which are not needed for BC

    trajectories.append(current_trajectory)
    current_trajectory = []

    # Convert the trajectories to the format required by bc_trainer
    expert_dataset = rollout.flatten_trajectories(trajectories)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        learning_rate=5e-4,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=1.0,
        device="cuda"
    )

    # Create a BC trainer and train the model
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=model.policy,
        # demonstrations=expert_dataset,  # FIXME generate expert_dataset
    )
    bc_trainer.train(n_epochs=100)

    model_name = 'expert-rocket-v0'
    model.save(model_name)


if __name__ == '__main__':
    train_and_evaluate()

