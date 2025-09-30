from gymnasium.envs.registration import register
from environments.windygrid import WindyGridEnv

register(
    id="environment/WindyGrid-v0",
    entry_point="environment.windygrid:WindyGridEnv",
    max_episode_steps=200
)