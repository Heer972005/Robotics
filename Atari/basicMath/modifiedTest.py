import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

env = gym.make(
    "ALE/BasicMath-v5",
    obs_type="grayscale",
    frameskip=1,          # 🔥 disable native frame-skip
    render_mode="human"
)

env = AtariPreprocessing(
    env,
    frame_skip=4,         # ✅ now safe
    grayscale_obs=True,
    scale_obs=True
)

env = FrameStackObservation(env, stack_size=4)

obs, info = env.reset()

print("Processed obs shape:", obs.shape)
print("dtype:", obs.dtype)
print("min/max:", obs.min(), obs.max())

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
