import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Environment
# -----------------------
env_id = "CartPole-v1"
env = gym.make(env_id)

s_size = env.observation_space.shape[0]
a_size = env.action_space.n

print("State size:", s_size)
print("Action size:", a_size)

# -----------------------
# Policy Network
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
        return action.item(), m.log_prob(action)

# -----------------------
# REINFORCE Algorithm
# -----------------------
def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    scores_deque = deque(maxlen=100)
    scores = []

    for episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []

        state, _ = env.reset()

        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)

            next_state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)

            if done or truncated:
                break

            state = next_state

        episode_return = sum(rewards)
        scores_deque.append(episode_return)
        scores.append(episode_return)

        # -----------------------
        # Compute discounted returns
        # -----------------------
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # -----------------------
        # Policy loss
        # -----------------------
        policy_loss = []
        for log_prob, Gt in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * Gt)

        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % print_every == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}")

    return scores

# -----------------------
# Hyperparameters
# -----------------------
h_size = 16
n_training_episodes = 1000
max_t = 1000
gamma = 1.0
lr = 1e-2

policy = Policy(s_size, a_size, h_size).to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)

# -----------------------
# Train
# -----------------------
scores = reinforce(
    policy,
    optimizer,
    n_training_episodes,
    max_t,
    gamma,
    print_every=100
)

# -----------------------
# Plot results
# -----------------------
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("REINFORCE on CartPole")
plt.show()

# -----------------------
# Save model
# -----------------------
torch.save(policy.state_dict(), "cartpole_reinforce.pth")
print("Model saved as cartpole_reinforce.pth")
