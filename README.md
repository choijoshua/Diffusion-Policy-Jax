# Diffusion-Policy-Jax

A JAX implementation of Generative Policies for imitation learning and offline reinforcement learning

## Overview
This repository implements generative models for policy learning, supporting both discrete-time DDPM and continuous-time score matching approaches. The implementation is designed for high-performance training with full JAX JIT compilation support.

## Features

### Hydra Configuration System
```
main_conf/
├── base_config.yaml          # Main config file
├── algorithms/
    ├── ddpm.yaml            # DDPM-specific parameters
    └── score_matching.yaml  # Score matching parameters
```

### Multiple Generative Policy Algorithms

| Algorithm | Config File |
| --- |--- |
| DDPM | [`config/algorithms/ddpm.yaml`](config/algorithms/ddpm.yaml) |
| Score Matching | [`config/algorithms/score_matching.yaml`](config/algorithms/score_matching.yaml) |

### Flexible Policy Modes
- Single Step Action Prediction
- Action Sequence Generation

### Architecture
- 1D UNet with time and observation conditioning
- Positional embeddings for temporal information
- Configurable depth and hidden dimensions

### Training Infrastructure
- Fully JIT-compatible training loop for maximum performance
- Support for D4RL and OGBench datasets
- Wandb integration for experiment tracking

## Basic Usage
```
# install dependencies
pip install -r requirements.txt

# Train on Hopper with DDPM policy
python train.py dataset=hopper-medium-v2 algorithm=ddpm

# Train with score matching
python train.py dataset=hopper-expert-v2 algorithm=score_matching

# Custom hyperparameters
python train.py dataset=hopper-medium-v2 \
    horizon=8 \
    batch_size=256 \
    lr=1e-4 \
    num_timesteps=100
```
