import gymnasium as gym
import panda_gym

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# Load environment
eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

eval_env.training = False
eval_env.norm_reward = False

# Load model
model = A2C.load("a2c-PandaReachDense-v3")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
