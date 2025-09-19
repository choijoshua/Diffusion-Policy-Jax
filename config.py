from dataclasses import dataclass

@dataclass
class Args:
    seed: int = 1
    dataset: str = "mujoco/halfcheetah/expert-v0"
    embed_dim: int = 100
    num_updates: int = 8
    num_residual_blocks: int = 4
    num_mid_blocks: int = 2
    dims: tuple = (32, 64, 128, 256)

