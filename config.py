from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Optional
from typing import Sequence

@dataclass
class TrainArgs:
    # --- Experiment ---
    algorithm: str = MISSING
    lr: float = 3e-4
    num_updates: int = 100000
    eval_interval: int = 1000
    eval_workers: int = 10
    batch_size: int = 128
    gamma: float = 0.99
    
    num_timesteps: int = 50
    mode: str = "trajectory"
    horizon: int = 64
    max_traj_len: int = 1000
    max_n_traj: int = 1000

    # Model Parameters
    embed_dim: int = 256
    dims: tuple = (128, 256, 512)

@dataclass
class DDPMArgs(TrainArgs):
    algorithm = "ddpm"

@dataclass
class ScoreMatchingArgs(TrainArgs):
    algorithm = "score_matching"
    sigma: float = 3.0
    eps: float = 1e-5
    sampler: str = "euler_maruyama" # euler_maruyama, predictor_corrector


@dataclass
class GeneralArgs:
    algorithms: TrainArgs = MISSING
    # --- Experiment ---
    seed: int = 0
    dataset: str = MISSING
    custom_dataset: Optional[str] = None
    # --- Logging ---
    log: bool = False
    save_ckpt: bool = False
    wandb_project: str = "unifloral"
    wandb_group: str = "debug"
    wandb_jobtype: str = "Train"

def register_train_configs() -> None:
    cs = ConfigStore.instance()
    
    cs.store(
        group="algorithms",
        name="ddpm",
        node=DDPMArgs,
    )
    cs.store(
        group="algorithms",
        name="score_matching",
        node=ScoreMatchingArgs,
    )