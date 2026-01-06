import gymnasium as gym
import numpy as np
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
    
env = OldGymWrapper(gym.make("CartPole-v1"))
states = env.observation_space.shape[0]
actions = env.action_space.n

model = Sequential([
    Flatten(input_shape=(1, states)),
    Dense(24, activation="relu"),
    Dense(24, activation="relu"),
    Dense(actions, activation="linear")
])

policy = BoltzmannQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)

dqn = DQNAgent(
    model=model,
    memory=memory,
    policy=policy,
    nb_actions=actions,
    enable_double_dqn=True
)

dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])
dqn.load_weights("ddqn_weights.h5")

dqn.test(env, nb_episodes=5, visualize=True)
env.close()
