import gymnasium as gym
import panda_gym
import time

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Create env with rendering
env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3", render_mode="human")])
env = VecNormalize.load("vec_normalize.pkl", env)

env.training = False
env.norm_reward = False

# Load trained model
model = A2C.load("a2c-PandaReachDense-v3")

obs = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    action = action * 0.3
    obs, reward, done, info = env.step(action)
    time.sleep(0.1)

    if done:
        obs = env.reset()
