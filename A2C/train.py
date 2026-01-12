import gymnasium as gym
import panda_gym

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

# ------------------------
# Environment
# ------------------------
env_id = "PandaReachDense-v3"

env = make_vec_env(env_id, n_envs=4)

# Normalize observation + reward
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# ------------------------
# Model
# ------------------------
model = A2C(
    policy="MultiInputPolicy",
    env=env,
    verbose=1
)

# ------------------------
# Training
# ------------------------
model.learn(total_timesteps=1_000_000)

# ------------------------
# Save
# ------------------------
model.save("a2c-PandaReachDense-v3")
env.save("vec_normalize.pkl")

print("Training completed and model saved.")
