import gymnasium as gym
import numpy as np
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
env = gym.make("CartPole-v1")
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

print("State size:", s_size)
print("Action size:", a_size)

# -----------------------
# Actor-Critic Network
# -----------------------
class ActorCritic(nn.Module):
    def __init__(self, s_size, a_size, h_size=128):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.actor = nn.Linear(h_size, a_size)
        self.critic = nn.Linear(h_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.actor(x), dim=1)
        state_value = self.critic(x)
        return action_probs, state_value


# -----------------------
# Initialize
# -----------------------
model = ActorCritic(s_size, a_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
gamma = 0.99
n_episodes = 1000
max_steps = 1000

scores = []   # <-- FOR GRAPH

# -----------------------
# Training Loop
# -----------------------
for episode in range(1, n_episodes + 1):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    episode_reward = 0

    for step in range(max_steps):
        probs, state_value = model(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, truncated, _ = env.step(action.item())
        episode_reward += reward

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        _, next_state_value = model(next_state_tensor)

        # -----------------------
        # Compute Advantage (NO .item())
        # -----------------------
        target_value = reward + gamma * next_state_value * (1 - int(done))
        advantage = target_value - state_value

        # -----------------------
        # Losses
        # -----------------------
        actor_loss = -log_prob * advantage.detach()
        critic_loss = F.mse_loss(state_value, target_value.detach())

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state_tensor

        if done or truncated:
            break

    scores.append(episode_reward)

    if episode % 50 == 0:
        print(f"Episode {episode} | Reward: {episode_reward}")

# -----------------------
# Plot results
# -----------------------
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Actor-Critic on CartPole")
plt.show()

# -----------------------
# Save model
# -----------------------
torch.save(model.state_dict(), "cartpole_actor_critic.pth")
print("Model saved as cartpole_actor_critic.pth")
