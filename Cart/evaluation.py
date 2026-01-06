import gymnasium as gym
import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Policy Network (same as training)
# -----------------------
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item()

# -----------------------
# Environment
# -----------------------
env = gym.make("CartPole-v1", render_mode="human")

s_size = env.observation_space.shape[0]
a_size = env.action_space.n

policy = Policy(s_size, a_size, h_size=16).to(device)
policy.load_state_dict(torch.load("cartpole_reinforce.pth", map_location=device))
policy.eval()

# -----------------------
# Evaluate
# -----------------------
n_eval_episodes = 10
max_steps = 1000

for ep in range(n_eval_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = policy.act(state)
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        if done or truncated:
            break

    print(f"Episode {ep+1} reward: {total_reward}")

env.close()
