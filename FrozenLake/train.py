import gymnasium as gym
from q_learning import train_q_learning
import pickle

env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=False,
    render_mode=None,
)

Qtable = train_q_learning(
    env=env,
    n_training_episodes=10000,
    learning_rate=0.7,
    gamma=0.95,
    max_steps=99,
    max_epsilon=1.0,
    min_epsilon=0.05,
    decay_rate=0.0005,
)

print("Training finished!")
print(Qtable)

with open("qtable.pkl", "wb") as f:
    pickle.dump(Qtable, f)

print("💾 Q-table saved as qtable.pkl")
