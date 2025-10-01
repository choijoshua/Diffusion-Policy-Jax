from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Optional

@dataclass
class TrainArgs:
    # --- Experiment ---
    algorithm: str = MISSING
    lr: float = 3e-4
    num_updates: int = 100000
    
    num_timesteps: int = 50
    

@dataclass
class DDPMArgs(TrainArgs):
    algorithm = "ddpm"

@dataclass
class ScoreMatchingArgs(TrainArgs):
    algorithm = "score_matching"
    sigma: float = 3.0


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
    wandb_team: str = "flair"
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