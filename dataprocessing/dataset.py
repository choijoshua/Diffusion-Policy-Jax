import jax
import numpy as onp
import jax.numpy as jnp
from dataprocessing.utils import Transition


class TrajDataset:
    """
    Dataset for sampling trajectory segments for diffusion policy training.
    Works directly with Transition namedtuples from build_transition_from_raw.
    """
    
    def __init__(self, dataset, horizon=64, max_traj_len=1000, max_n_traj=1000):
        """
        Args:
            dataset: Transition namedtuple with fields:
                    'obs', 'action', 'reward', 'next_obs', 'next_action', 'done', 'traj_return'
                    Each field is a JAX/numpy array of shape (N, ...)
            horizon: Length of trajectory segments to sample
            max_traj_len: Maximum trajectory length (for memory allocation)
            max_n_traj: Maximum number of trajectories
        """
        self.horizon = horizon
        self.max_traj_len = max_traj_len
        self.max_n_traj = max_n_traj
        
        # Process the Transition namedtuple
        self._process_transition_dataset(dataset)
    
    def _process_transition_dataset(self, dataset):
        """
        Process Transition namedtuple into trajectories using 'done' flags.
        
        Args:
            dataset: Transition namedtuple with flat arrays
        """
        # Convert JAX arrays to numpy for indexing (JAX is immutable)
        obs = onp.array(dataset.obs)
        actions = onp.array(dataset.action)
        rewards = onp.array(dataset.reward)
        next_obs = onp.array(dataset.next_obs)
        next_actions = onp.array(dataset.next_action)
        dones = onp.array(dataset.done).astype(bool)
        traj_returns = onp.array(dataset.traj_return)
        
        # Find episode boundaries using 'done' flags
        # done=True/1 indicates the end of an episode
        done_indices = onp.where(dones)[0]  # Indices where done==True
        
        if len(done_indices) == 0:
            # No episode boundaries found, treat entire dataset as one episode
            print("Warning: No 'done' flags found. Treating entire dataset as one trajectory.")
            done_indices = onp.array([len(dones) - 1])
        
        # Episode boundaries: starts at 0 or after previous done, ends at done index
        # starts = [0, done_idx1+1, done_idx2+1, ...]
        # ends = [done_idx1+1, done_idx2+1, ...]  (+1 to include the terminal state)
        self.starts = onp.concatenate([[0], done_indices[:-1] + 1])
        self.ends = done_indices + 1
        
        # Store the full concatenated arrays
        self.obs = obs
        self.acts = actions
        self.rews = rewards
        self.next_obs = next_obs
        self.next_acts = next_actions
        self.dones = dones
        self.traj_returns = traj_returns
        
        num_trajectories = len(self.ends)
        total_transitions = len(obs)
        mean_length = onp.mean(self.ends - self.starts)
        
        print(f"Dataset processed: {num_trajectories} trajectories, "
              f"{total_transitions} total transitions")
        print(f"Mean trajectory length: {mean_length:.1f}, "
              f"Min: {onp.min(self.ends - self.starts)}, "
              f"Max: {onp.max(self.ends - self.starts)}")
    
    def sample_batch(self, rng, batch_size):
        """
        Sample a batch of trajectory segments deterministically using JAX RNG.
        
        Args:
            rng: JAX random key
            batch_size: Number of trajectory segments to sample
        
        Returns:
            new_rng: Updated JAX random key
            batch: Tuple of (obs_batch, act_batch, rew_batch, done_batch, 
                            next_obs_batch, next_act_batch, rtg_batch)
        """
        obs_batch = []
        act_batch = []
        rew_batch = []
        done_batch = []
        next_obs_batch = []
        next_act_batch = []
        rtg_batch = []
        
        for i in range(batch_size):
            # Split RNG for this sample
            rng, ep_rng, segment_rng = jax.random.split(rng, 3)
            
            # Pick an episode uniformly using JAX random
            ep_idx = jax.random.randint(ep_rng, (), 0, len(self.ends))
            ep_start = self.starts[ep_idx]
            ep_end = self.ends[ep_idx]
            
            # Episode length
            ep_length = ep_end - ep_start
            
            if ep_length >= self.horizon:
                # Sample uniformly from valid range using JAX random
                end = jax.random.randint(segment_rng, (), ep_start + self.horizon, ep_end + 1)
                start = end - self.horizon
                pad = 0
            else:
                # Episode shorter than horizon - use entire episode and pad
                start = ep_start
                end = ep_end
                pad = self.horizon - (end - start)
            
            # Fetch slices
            idxs = onp.arange(start, end)
            o = self.obs[idxs]          # [L, obs_dim]
            a = self.acts[idxs]         # [L, action_dim]
            r = self.rews[idxs]         # [L]
            d = self.dones[idxs]        # [L]
            no = self.next_obs[idxs]    # [L, obs_dim]
            na = self.next_acts[idxs]   # [L, action_dim]
            rtg = self.traj_returns[idxs]  # [L]
            
            # Apply padding if needed (pad at the front)
            if pad > 0:
                # Pad by repeating first frame for observations
                o = onp.vstack([onp.repeat(o[:1], pad, axis=0), o])
                no = onp.vstack([onp.repeat(no[:1], pad, axis=0), no])
                # Pad with zeros for actions
                a = onp.vstack([onp.zeros((pad, a.shape[1])), a])
                na = onp.vstack([onp.zeros((pad, na.shape[1])), na])
                # Pad with zeros for rewards, dones, and returns
                r = onp.concatenate([onp.zeros(pad), r])
                d = onp.concatenate([onp.zeros(pad, dtype=bool), d])
                rtg = onp.concatenate([onp.zeros(pad), rtg])
            
            obs_batch.append(o)
            act_batch.append(a)
            rew_batch.append(r)
            done_batch.append(d)
            next_obs_batch.append(no)
            next_act_batch.append(na)
            rtg_batch.append(rtg)
        
        # Stack into batches and convert to JAX arrays
        obs_batch = jnp.array(onp.stack(obs_batch))       # [B, T, obs_dim]
        act_batch = jnp.array(onp.stack(act_batch))       # [B, T, action_dim]
        rew_batch = jnp.array(onp.stack(rew_batch))       # [B, T]
        done_batch = jnp.array(onp.stack(done_batch))     # [B, T]
        next_obs_batch = jnp.array(onp.stack(next_obs_batch))  # [B, T, obs_dim]
        next_act_batch = jnp.array(onp.stack(next_act_batch))  # [B, T, action_dim]
        rtg_batch = jnp.array(onp.stack(rtg_batch))       # [B, T]
        
        return (obs_batch, act_batch, rew_batch, done_batch, 
                    next_obs_batch, next_act_batch, rtg_batch)
    
    def __len__(self):
        """Return number of trajectories."""
        return len(self.ends)
    
    def get_stats(self):
        """Get dataset statistics."""
        traj_lengths = self.ends - self.starts
        episode_returns = []
        
        for i in range(len(self.ends)):
            start, end = self.starts[i], self.ends[i]
            # Use the first traj_return value (which is the total return for the episode)
            episode_return = self.traj_returns[start]
            episode_returns.append(episode_return)
        
        return {
            'num_trajectories': len(self.ends),
            'total_transitions': len(self.obs),
            'obs_dim': self.obs.shape[1] if len(self.obs.shape) > 1 else 1,
            'action_dim': self.acts.shape[1] if len(self.acts.shape) > 1 else 1,
            'mean_traj_length': float(onp.mean(traj_lengths)),
            'std_traj_length': float(onp.std(traj_lengths)),
            'min_traj_length': int(onp.min(traj_lengths)),
            'max_traj_length': int(onp.max(traj_lengths)),
            'mean_episode_return': float(onp.mean(episode_returns)),
            'std_episode_return': float(onp.std(episode_returns)),
            'min_episode_return': float(onp.min(episode_returns)),
            'max_episode_return': float(onp.max(episode_returns)),
        }

