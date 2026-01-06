import gymnasium as gym
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class OldGymWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

# -------------------------
# Create Environment
# -------------------------
env = OldGymWrapper(gym.make("CartPole-v1"))
states = env.observation_space.shape[0]
actions = env.action_space.n

print("States:", states)
print("Actions:", actions)

# -------------------------
# Build Deep Q Network
# -------------------------
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model

model = build_model(states, actions)
model.summary()

# -------------------------
# Build Double DQN Agent
# -------------------------
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)

    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10,
        gamma=0.99,
        target_model_update=1e-2,
        enable_double_dqn=True
    )
    return dqn

dqn = build_agent(model, actions)

dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])

# -------------------------
# Train Agent
# -------------------------
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# -------------------------
# Test Agent
# -------------------------
scores = dqn.test(env, nb_episodes=10, visualize=False)
print("Average Reward:", np.mean(scores.history["episode_reward"]))

# -------------------------
# Save Weights
# -------------------------
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
dqn.save_weights("ddqn_weights.h5", overwrite=True)
dqn.load_weights("ddqn_weights.h5")
env.close()
