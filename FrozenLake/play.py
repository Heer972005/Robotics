import gymnasium as gym
import time
from q_learning import greedy_policy
import numpy as np
import pickle

# Load Q-table
with open("qtable.pkl", "rb") as f:
    Qtable = pickle.load(f)

# Create environment with human rendering
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False,
    render_mode="human"
)

state, _ = env.reset()
done = False

print("🎮 Agent is playing FrozenLake...")

while not done:
    action = greedy_policy(Qtable, state)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    time.sleep(1.5)  # slow down for visibility

if reward == 1:
    print("🏆 Agent reached the goal!")
else:
    print("❌ Agent failed!")

env.close()
