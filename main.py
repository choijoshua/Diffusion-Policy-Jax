import jax
import jax.numpy as jnp
import tyro
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import minari
from config import Args
from diffusion import DiffusionPolicy
from model import UNet
from train import create_train_state, make_train_step
from collections import namedtuple


Transition = namedtuple("Transition", "obs action reward next_obs next_action done traj_return label")


if __name__=="__main__":
    args = tyro.cli(Args)
    rng = jax.random.PRNGKey(args.seed)
    raw_dataset = minari.load_dataset(args.dataset, download=True)

    def make_env():
        return raw_dataset.recover_environment()

    num_envs = 8
    env_fns  = [make_env for _ in range(num_envs)]

    env = SyncVectorEnv(env_fns)

    dataset_list = []
    for episode in raw_dataset:
        episode_data = Transition(
                obs=jnp.array(episode.observations[:-1]),
                action=jnp.array(episode.actions),
                reward=jnp.array(episode.rewards),
                next_obs=jnp.array(episode.observations[1:]),
                next_action=jnp.roll(episode.actions, -1, axis=0),
                done=jnp.logical_or(jnp.array(episode.truncations), jnp.array(episode.terminations)),
                traj_return=jnp.zeros_like(jnp.array(episode.truncations)),
                label=jnp.zeros_like(jnp.array(episode.truncations))
        )
        dataset_list.append(episode_data)

    # dataset = jax.tree_map(lambda *arrays: jnp.concatenate(arrays, axis=0), *dataset_list)
    act_dim = env.single_action_space.shape[0]
    obs_dim = env.single_observation_space.shape[0]
    dummy_obs = jnp.zeros(env.single_observation_space.shape)
    dummy_action = jnp.zeros(act_dim)
    network = UNet(args, obs_dim, act_dim)

    agent_state = create_train_state(args, rng, )

