import jax
import jax.numpy as jnp
import numpy as np
from dataprocessing.utils import Transition


class TrajDataset:
    """JAX-compatible trajectory dataset for use inside JIT."""
    
    def __init__(self, dataset, horizon=64, max_traj_len=1000, max_n_traj=1000):
        """
        Args:
            dataset: Transition namedtuple with fields:
                    'obs', 'action', 'reward', 'next_obs', 'next_action', 'done', 'traj_return'
            horizon: Length of trajectory segments to sample
            max_traj_len: Maximum trajectory length
            max_n_traj: Maximum number of trajectories
        """
        self.horizon = horizon
        self.max_traj_len = max_traj_len
        self.max_n_traj = max_n_traj
        
        self._process_transition_dataset(dataset)
    
    def _process_transition_dataset(self, dataset):
        """Convert to JAX arrays and compute episode boundaries."""
        max_n_traj = self.max_n_traj
        # Convert to numpy for processing
        obs = np.array(dataset.obs)
        actions = np.array(dataset.action)
        rewards = np.array(dataset.reward)
        next_obs = np.array(dataset.next_obs)
        next_actions = np.array(dataset.next_action)
        dones = np.array(dataset.done).astype(bool)
        traj_returns = np.array(dataset.traj_return)
        
        # Find episode boundaries
        done_indices = np.where(dones)[0]
        if len(done_indices) == 0:
            print("Warning: No 'done' flags found. Treating entire dataset as one trajectory.")
            done_indices = np.array([len(dones) - 1])
        
        num_episodes = len(done_indices)
        starts = np.concatenate([[0], done_indices[:-1] + 1])
        ends = done_indices + 1
        
        # Pad episode boundaries to max_n_traj for fixed-size arrays
        if num_episodes < max_n_traj:
            pad_size = max_n_traj - num_episodes
            starts = np.concatenate([starts, np.zeros(pad_size, dtype=np.int32)])
            ends = np.concatenate([ends, np.full(pad_size, len(obs), dtype=np.int32)])
        else:
            starts = starts[:max_n_traj]
            ends = ends[:max_n_traj]
            num_episodes = max_n_traj
        
        # Store as JAX arrays
        self.obs = jnp.array(obs)
        self.acts = jnp.array(actions)
        self.rews = jnp.array(rewards)
        self.next_obs = jnp.array(next_obs)
        self.next_acts = jnp.array(next_actions)
        self.dones = jnp.array(dones)
        self.traj_returns = jnp.array(traj_returns)
        
        self.starts = jnp.array(starts)
        self.ends = jnp.array(ends)
        self.num_episodes = num_episodes
        
        mean_length = np.mean(ends[:num_episodes] - starts[:num_episodes])
        print(f"Dataset processed: {num_episodes} trajectories, {len(obs)} total transitions")
        print(f"Mean trajectory length: {mean_length:.1f}, "
              f"Min: {np.min(ends[:num_episodes] - starts[:num_episodes])}, "
              f"Max: {np.max(ends[:num_episodes] - starts[:num_episodes])}")
    
    def _sample_single_segment(self, rng, ep_idx):
        """Sample a single trajectory segment (fully JAX-compatible)."""
        ep_start = self.starts[ep_idx]
        ep_end = self.ends[ep_idx]
        
        # Sample segment start position
        rng, seg_rng = jax.random.split(rng)
        
        # Compute valid range for sampling
        # If episode is shorter than horizon, start at ep_start
        # If episode is longer, sample uniformly
        max_start = jnp.maximum(ep_start, ep_end - self.horizon)
        start = jax.random.randint(seg_rng, (), ep_start, max_start + 1)
        
        # Always extract exactly horizon-length slices (FIXED SIZE for JAX)
        obs_seg = jax.lax.dynamic_slice(
            self.obs,
            (start, 0),
            (self.horizon, self.obs.shape[1])
        )
        
        act_seg = jax.lax.dynamic_slice(
            self.acts,
            (start, 0),
            (self.horizon, self.acts.shape[1])
        )
        
        rew_seg = jax.lax.dynamic_slice(
            self.rews,
            (start,),
            (self.horizon,)
        )
        
        done_seg = jax.lax.dynamic_slice(
            self.dones,
            (start,),
            (self.horizon,)
        )
        
        next_obs_seg = jax.lax.dynamic_slice(
            self.next_obs,
            (start, 0),
            (self.horizon, self.next_obs.shape[1])
        )
        
        next_act_seg = jax.lax.dynamic_slice(
            self.next_acts,
            (start, 0),
            (self.horizon, self.next_acts.shape[1])
        )
        
        rtg_seg = jax.lax.dynamic_slice(
            self.traj_returns,
            (start,),
            (self.horizon,)
        )
        
        # Create mask for valid data
        # If we sampled past episode end, mask out invalid data
        end = jnp.minimum(start + self.horizon, ep_end)
        actual_length = end - start
        mask = jnp.arange(self.horizon) < actual_length
        
        # Apply mask: zero out or repeat for padding
        # For observations, repeat first valid frame in padding region
        obs_seg = jnp.where(mask[:, None], obs_seg, obs_seg[0:1])
        act_seg = jnp.where(mask[:, None], act_seg, 0.0)
        rew_seg = jnp.where(mask, rew_seg, 0.0)
        done_seg = jnp.where(mask, done_seg, False)
        next_obs_seg = jnp.where(mask[:, None], next_obs_seg, next_obs_seg[0:1])
        next_act_seg = jnp.where(mask[:, None], next_act_seg, 0.0)
        rtg_seg = jnp.where(mask, rtg_seg, 0.0)
        
        return obs_seg, act_seg, rew_seg, done_seg, next_obs_seg, next_act_seg, rtg_seg
    
    def sample_batch(self, rng, batch_size):
        """
        JAX-compatible batch sampling using vmap.
        Can be used inside JIT/scan.
        
        Args:
            rng: JAX random key
            batch_size: Number of trajectory segments to sample
        
        Returns:
            tuple of (obs_batch, act_batch, rew_batch, done_batch, 
                     next_obs_batch, next_act_batch, rtg_batch)
            Each with shape (batch_size, horizon, ...)
        """
        # Sample episode indices
        rng, ep_rng = jax.random.split(rng)
        ep_indices = jax.random.randint(ep_rng, (batch_size,), 0, self.num_episodes)
        
        # Generate RNG keys for each sample
        rngs = jax.random.split(rng, batch_size)
        
        # Use vmap to sample all trajectories in parallel
        batch_data = jax.vmap(self._sample_single_segment)(rngs, ep_indices)
        
        return batch_data
    
    def __len__(self):
        """Return number of trajectories."""
        return self.num_episodes
    
    def get_stats(self):
        """Get dataset statistics (called outside JIT)."""
        traj_lengths = self.ends[:self.num_episodes] - self.starts[:self.num_episodes]
        episode_returns = self.traj_returns[self.starts[:self.num_episodes]]
        
        return {
            'num_trajectories': self.num_episodes,
            'total_transitions': len(self.obs),
            'obs_dim': int(self.obs.shape[1]),
            'action_dim': int(self.acts.shape[1]),
            'mean_traj_length': float(jnp.mean(traj_lengths)),
            'std_traj_length': float(jnp.std(traj_lengths)),
            'min_traj_length': int(jnp.min(traj_lengths)),
            'max_traj_length': int(jnp.max(traj_lengths)),
            'mean_episode_return': float(jnp.mean(episode_returns)),
            'std_episode_return': float(jnp.std(episode_returns)),
            'min_episode_return': float(jnp.min(episode_returns)),
            'max_episode_return': float(jnp.max(episode_returns)),
        }