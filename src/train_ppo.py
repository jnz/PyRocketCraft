import gymasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from simrocketenv import SimRocketEnv

def make_env():
    def _init():
        env = SimRocketEnv(interactive=False)
        return env
    return _init

def train_and_evaluate():
    num_envs = 1
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        learning_rate=6e-4,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.04,
        vf_coef=0.5,
        max_grad_norm=1.0,
        device="cuda"
    )

    model.learn(total_timesteps=9000000)
    model_name = 'ppo-rocket-v0'
    model.save(model_name)

    # eval_env = SimRocketEnv(interactive=False)
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

if __name__ == '__main__':
    train_and_evaluate()
