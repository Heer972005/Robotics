import gymnasium as gym
import panda_gym

env = gym.make("PandaReachDense-v3")
obs, info = env.reset()

print("Observation:", obs)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
