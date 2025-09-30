import gymnasium as gym
import environments 
import numpy as np
from stable_baselines3 import PPO

env = gym.make("environment/WindyGrid-v0")
model = PPO(
    policy="MlpPolicy",   # fully connected net
    env=env,
    verbose=1,
    batch_size=64,
    learning_rate=3e-4,
)
model.learn(total_timesteps=500000)

# 5) Test the trained policy
obs, _ = env.reset()
for i in range(120):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1:2d} → pos={obs}, reward={reward}, truncated={truncated}, info={info}")
    if terminated or truncated:
        break


