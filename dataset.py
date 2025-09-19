import numpy as onp
import jax.numpy as jnp

class TrajDataset:
    def __init__(self, dataset_list=[], horizon=64, max_traj_len=1000, max_n_traj=1000):
        self.horizon = horizon
        self.dataset = dataset_list
        self.max_traj_len = max_traj_len
        self.max_n_traj = max_n_traj

    def sample_batch(self, batch_size):
        obs_batch  = []
        act_batch  = []
        rew_batch  = []
        done_batch = []

        for _ in range(batch_size):
            # pick an episode end uniformly,
            # then choose a start T steps *before* it
            end = int(np.random.choice(self.ends))
            start = max(0, end - self.horizon)
            # if episode is shorter than horizon, just pad at the front:
            pad = self.horizon - (end - start)
            idxs = np.arange(start, end)
            # fetch slices
            o = self.obs[idxs]       # [L, obs_dim]
            a = self.acts[idxs]      # [L, action_dim]
            r = self.rews[idxs]      # [L]
            d = self.dones[idxs]     # [L]
            if pad > 0:
                # pad with zeros (or repeat first frame)
                o = np.vstack([o[:1]]*pad + [o])
                a = np.vstack([np.zeros_like(a[:1])]*pad + [a])
                r = np.concatenate([np.zeros(pad), r])
                d = np.concatenate([np.zeros(pad), d])
            obs_batch .append(o)  # now shape [horizon, obs_dim]
            act_batch .append(a)
            rew_batch .append(r)
            done_batch.append(d)

        obs_batch  = np.stack(obs_batch)   # [B, T, obs_dim]
        act_batch  = np.stack(act_batch)   # [B, T, action_dim]
        rew_batch  = np.stack(rew_batch)   # [B, T]
        done_batch = np.stack(done_batch)  # [B, T]

        # now concatenate into the “diffusion” format:
        # [obs,action,reward,done] at each step

        return obs_batch, act_batch
