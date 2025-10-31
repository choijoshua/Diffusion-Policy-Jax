from gymnasium.envs.registration import register
from environments.windygrid import WindyGridEnv
from environments.quad_env import QuadrotorEnv

register(
    id="environment/WindyGrid-v0",
    entry_point="environment.windygrid:WindyGridEnv",
    max_episode_steps=200
)

register(
    id="environment/QuadEnv-v0",
    entry_point="environment.quad_env:QuadrotorEnv",
    max_episode_steps=1000
)