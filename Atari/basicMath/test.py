import gymnasium as gym
import ale_py   # ⭐ THIS LINE REGISTERS ALE ENVIRONMENTS ⭐

env = gym.make(
    "ALE/BasicMath-v5",
    obs_type="grayscale",
    render_mode="human"
)

obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Observation dtype:", obs.dtype)
print("Min/Max:", obs.min(), obs.max())

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
