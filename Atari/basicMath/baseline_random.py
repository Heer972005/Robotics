import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import numpy as np

env = gym.make(
    "ALE/BasicMath-v5",
    obs_type="grayscale",
    frameskip=1
)

env = AtariPreprocessing(
    env,
    frame_skip=4,
    grayscale_obs=True,
    scale_obs=True
)

env = FrameStackObservation(env, stack_size=4)

NUM_EPISODES = 20
episode_rewards = []

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    last_action = None

    while not done:
        if last_action == 1:   # FIRE
            action = np.random.choice([0,2,3,4,5])
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        last_action = action
        done = terminated or truncated

    episode_rewards.append(total_reward)
    print(f"Episode {ep+1}: reward = {total_reward}")

env.close()

print("\nAverage disciplined random reward:", np.mean(episode_rewards))
