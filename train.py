import jax
import jax.numpy as jnp
import numpy as onp
import optax
from flax.training.train_state import TrainState
import hydra
from hydra.core.config_store import ConfigStore
import sys
from pathlib import Path
import pickle
import warnings
import wandb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataprocessing.utils import make_env_and_dataset, build_transition_from_raw, Transition
from dataprocessing.dataset import TrajDataset
from config import GeneralArgs, TrainArgs, register_train_configs
from algorithms.ddpm import DDPMPolicy
from algorithms.score_matching import ScoreMatchingPolicy
from model.unet import UNet
from collections import namedtuple


def create_train_state(args, rng, network, dummy_input, steps=None):
    lr = optax.cosine_decay_schedule(args.lr, steps or args.num_updates)
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(lr, eps=1e-5),
    )

def make_train_step(args, dataset):
    """
    Create train step that samples from TrajDataset with deterministic RNG.
    
    Args: 
        args: TrainArgs for algorithm
        dataset: dataset of type TrajDataset
    """
    
    def _train_step(runner_state, _):
        rng, policy = runner_state

        # Define loss function
        def ddpm_loss_fn(params):
            eps_pred = policy.predict(params, x_t, t, batch_obs)
            loss = jnp.mean((noise - eps_pred) ** 2)
            return loss
        
        def score_matching_loss_fn(params):
            score_pred = policy.predict(params, x_t, t, batch_obs)
            std = policy.forward_sde_variance(t)
            loss = jnp.mean(jnp.sum((score_pred * std[:, None, None] + noise)**2, 
                          axis=(1,2,3)))
            return loss
        
        # 1) Split RNG for different operations
        rng, sample_rng, t_rng, noise_rng = jax.random.split(rng, 4)
        
        # 2) Sample batch from trajectory dataset using RNG
        batch = dataset.sample_batch(sample_rng, args.batch_size)
        obs_batch, act_batch, rew_batch, done_batch, next_obs_batch, next_act_batch, rtg_batch = batch
        
        if args.mode == 'single':
            # Single-step diffusion: denoise single actions
            timestep_idx = 0
            batch_x0 = act_batch[:, timestep_idx, :]    # (B, action_dim)
            batch_obs = obs_batch[:, timestep_idx, :]   # (B, obs_dim)
            
        elif args.mode == 'trajectory':
            # Trajectory diffusion: denoise entire action sequence
            batch_x0 = act_batch  # (B, horizon, action_dim)
            batch_obs = obs_batch[:, 0, :]  # Condition on initial observation

        # 3) Sample diffusion timesteps
        if args.algorithms == "ddpm":
            t = jax.random.randint(t_rng, (args.batch_size,), 0, args.num_timesteps)
            loss_fn = ddpm_loss_fn
        elif args.algorithms == "score_matching":
            t = jax.random.randint(t_rng, (args.batch_size,), args.eps, 1.0)
            loss_fn = score_matching_loss_fn
        else:
            print("CHOOSE AN ALGORITHM TO USE (DDPM, SCOREMATCHING)")
            return
        
        # 4) Sample noise and create noisy samples
        noise = jax.random.normal(noise_rng, batch_x0.shape)
        x_t = policy.forward_sample(batch_x0, t, noise)
        
        # 6) Compute gradients and update
        loss, grads = jax.value_and_grad(loss_fn)(policy.model.params)
        policy.model = policy.model.apply_gradients(grads=grads)
        
        # 7) Return new state and metrics
        new_runner_state = (rng, policy)
        metrics = {'loss': loss}
        return new_runner_state, metrics
    
    return jax.jit(_train_step)

def evaluate_policy_returns(
    cfg, 
    rng, 
    env, 
    policy, 
    env_meta=None):
    args = cfg.algorithms
    if env_meta is None:
        # Backward-compat: behave like before (raw returns)
        env_meta = {"kind": "custom", "make_single_unwrapped": lambda: env.env_fns[0]()}
    
    # Single underlying env (native type for that track)
    single_env = env_meta["make_single_unwrapped"]()
    # Episode length
    max_episode_steps = getattr(getattr(single_env, "spec", None), "max_episode_steps", None)
    if max_episode_steps is None:
        max_episode_steps = getattr(cfg, "max_episode_steps", 1000)

    # --- Reset environment ---
    step = 0
    returned = onp.zeros(args.eval_workers).astype(bool)
    cum_reward = onp.zeros(args.eval_workers)
    success = onp.zeros(args.eval_workers).astype(int)
    obs, _ = env.reset()

    # --- Rollout agent ---
    @jax.jit
    @jax.vmap
    def _policy_step(rng, obs):
        action = policy.get_action(rng, obs)
        return jnp.nan_to_num(action)

    while step < max_episode_steps and not returned.all():
        # --- Take step in environment ---
        step += 1
        rng, rng_step = jax.random.split(rng)
        rng_step = jax.random.split(rng_step, args.eval_workers)
        action = _policy_step(rng_step, jnp.array(obs))
        obs, reward, truncation, termination, info = env.step(onp.array(action))
        done = onp.logical_or(truncation, termination)
        # --- Track cumulative reward ---
        cum_reward += reward * ~returned
        returned |= done
        success |= truncation

    if step >= max_episode_steps and not returned.all():
        warnings.warn("Maximum steps reached before all episodes terminated")
    
    # Branch by dataset kind
    if env_meta["kind"] == "d4rl":
        import d4rl
        scores = d4rl.get_normalized_score(cfg.dataset, cum_reward) * 100.0
        return onp.mean(scores), onp.std(scores)
    else:
        # OGBench or custom: raw returns
        return onp.mean(success*100), onp.std(success*100)
    

cs = ConfigStore.instance()
register_train_configs()
cs.store(name="base_config", node=GeneralArgs)

@hydra.main(version_base=None, config_path="../main_conf", config_name="base_config")
def main(cfg: GeneralArgs) -> None:
    train_args = cfg.algorithms
    print(cfg)
    rng = jax.random.PRNGKey(cfg.seed)

    if cfg.log:
        wandb.init(
            config=train_args,
            project=cfg.wandb_project,
            # entity=args.wandb_team,
            group=cfg.wandb_group,
            job_type=cfg.wandb_jobtype,
        )

    # --- Initialize environment and dataset ---
    env, raw_dataset, env_meta = make_env_and_dataset(cfg, train_args.eval_workers)
    dataset = build_transition_from_raw(raw_dataset, train_args.gamma)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Dataset size: {len(dataset['obs'])}")   

    # --- Create TrajDataset from Transition namedtuple ---
    print("\nCreating trajectory dataset...")
    traj_dataset = TrajDataset(
        dataset=dataset,
        horizon=train_args.horizon,
        max_traj_len=train_args.max_traj_len,
        max_n_traj=train_args.max_n_traj
    )
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    stats = traj_dataset.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # --- Initialize UNet --- 
    print("\nInitializing UNet...")
    rng, dummy_rng, init_rng = jax.random.split(rng, 3)
    unet = UNet(
        args=train_args, action_dim=action_dim
    )
    x = jax.random.normal(dummy_rng, (train_args.batch_size, train_args.horizon, action_dim))  # [batch, horizon, action_dim]
    t = jnp.arange(train_args.batch_size)  # [batch,]
    obs = jax.random.normal(rng, (train_args.batch_size, obs_dim))  # [batch, obs_dim]
    policy_train_state = create_train_state(train_args, init_rng, unet, [x, t, obs], train_args.num_updates)

    if train_args.algorithm == "ddpm":
        policy = DDPMPolicy(
            train_args, policy_train_state, action_dim
        )
    elif train_args.algorithm == "score_matching":
        policy = ScoreMatchingPolicy(
            train_args, policy_train_state, action_dim
        )
    else: 
        print("CHOOSE AN ALGORITHM TO USE")
        return
    
    _agent_train_step_fn = make_train_step(
        train_args,
        dataset
    )
    
    num_evals = train_args.num_updates // train_args.eval_interval
    best_score = -onp.inf
    for eval_idx in range(num_evals):
        # --- Execute train loop ---
        (rng, policy), loss = jax.lax.scan(
            _agent_train_step_fn,
            (rng, policy),
            None,
            train_args.eval_interval,
        )

        # --- Evaluate agent ---
        rng, rng_eval = jax.random.split(rng)
        scores, std = evaluate_policy_returns(cfg, rng_eval, env, policy, env_meta)
        step = (eval_idx + 1) * train_args.eval_interval
        print("Step:", step, f"\t Score: {scores.mean():.2f}")
        if cfg.log:
            log_dict = {
                "score": scores.mean(),
                "score_std": scores.std(),
                "num_updates": step,
                **{k: loss[k][-1] for k in loss},
            }
            wandb.log(log_dict)

        