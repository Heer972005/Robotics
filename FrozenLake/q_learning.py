import numpy as np
import random

def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))


def greedy_policy(Qtable, state):
    return np.argmax(Qtable[state])


def epsilon_greedy_policy(Qtable, state, epsilon, action_space):
    if random.uniform(0, 1) > epsilon:
        return greedy_policy(Qtable, state)
    else:
        return random.randint(0, action_space - 1)


def train_q_learning(
    env,
    n_training_episodes,
    learning_rate,
    gamma,
    max_steps,
    max_epsilon,
    min_epsilon,
    decay_rate,
):
    state_space = env.observation_space.n
    action_space = env.action_space.n

    Qtable = initialize_q_table(state_space, action_space)

    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode
        )

        state, _ = env.reset()

        for _ in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon, action_space)
            new_state, reward, terminated, truncated, _ = env.step(action)

            Qtable[state, action] += learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state, action]
            )

            if terminated or truncated:
                break

            state = new_state

    return Qtable
