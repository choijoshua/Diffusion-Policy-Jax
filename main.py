import jax
import jax.numpy as jnp
import hydra
from hydra.core.config_store import ConfigStore
import sys
from pathlib import Path
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataprocessing.utils import make_env_and_dataset, build_transition_from_raw
import environments
from config import GeneralArgs, TrainArgs, register_train_configs
from algorithms.ddpm import DDPMPolicy
from model.unet import UNet
from train.train import create_train_state, make_train_step
from collections import namedtuple


Transition = namedtuple("Transition", "obs action reward next_obs next_action done traj_return label")

cs = ConfigStore.instance()
register_train_configs()
cs.store(name="base_config", node=GeneralArgs)

@hydra.main(version_base=None, config_path="../main_conf", config_name="base_config")
def main(cfg: GeneralArgs) -> None:
    train_args = cfg.algorithms
    print(cfg)
    rng = jax.random.PRNGKey(cfg.seed)

    # --- Initialize environment and dataset ---
    env, raw_dataset, env_meta = make_env_and_dataset(cfg, train_args.eval_workers)
    dataset = build_transition_from_raw(raw_dataset, train_args.gamma)



