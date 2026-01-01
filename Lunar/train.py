import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# 1. Create environment
env = Monitor(gym.make("LunarLander-v2"))

# 2. Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.001, 
    verbose=1
)

# 3. Train
model.learn(total_timesteps=100_000)

# 4. Save model
model.save("ppo_lunar")

env.close()
print("Training finished")