import gymnasium as gym
import numpy as np
from q_learning import greedy_policy

def evaluate_agent(env, Q, n_eval_episodes=100, max_steps=99):
    rewards = []

    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = greedy_policy(Q, state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
        render_mode=None,
    )

    from train import Qtable

    mean, std = evaluate_agent(env, Qtable)
    print(f"Mean reward: {mean:.2f} ± {std:.2f}")
