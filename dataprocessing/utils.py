from absl import flags
from collections import namedtuple
import numpy as onp
import jax.numpy as jnp
import jax
import gym as gym_legacy
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import pickle
from dataprocessing.d4rl_wrapper import ClassicGymToGymnasium
Transition = namedtuple("Transition", "obs action reward next_obs next_action done traj_return label")
FLAGS = flags.FLAGS


def split_into_trajectories(dataset, gamma):
    # 1) Pull everything out as NumPy once
    obs_np         = onp.array(dataset.obs)
    action_np      = onp.array(dataset.action)
    reward_np      = onp.array(dataset.reward)
    next_obs_np    = onp.array(dataset.next_obs)
    next_action_np = onp.array(dataset.next_action)
    done_np        = onp.array(dataset.done).astype(bool)
    label_np       = onp.array(dataset.label)

    # 2) Compute a flat return‐to‐go array of length N
    N = reward_np.shape[0]
    returns_np = onp.zeros_like(reward_np)
    done_idxs  = onp.nonzero(done_np)[0].tolist()

    start = 0
    for end in done_idxs:
        r_seg = reward_np[start:end+1]          # shape (T_i,)

        # method A: explicit vectorized
        T_i      = r_seg.shape[0]
        discounts = gamma**onp.arange(T_i)       # [1, γ, γ^2, … γ^(T_i-1)]
        # for each t, G_t = sum_{j=t..T_i-1} γ^(j-t) · r_seg[j]
        #          = ( discounts[:T_i-t] * r_seg[t:] ).sum()
        # we can build a toeplitz‐like operation, but simplest is:
        ret_seg = onp.array([
            onp.dot(discounts[:T_i - t], r_seg[t:]) 
            for t in range(T_i)
        ])

        returns_np[start:end+1] = ret_seg
        start = end + 1

    # 3) Build the per-episode list
    dataset_traj = []
    start = 0
    for end in done_idxs:
        # slice out Python‐numpy views
        o   = obs_np        [start:end+1]
        a   = action_np     [start:end+1]
        r   = reward_np     [start:end+1]
        no  = next_obs_np   [start:end+1]
        na  = next_action_np[start:end+1]
        d   = done_np       [start:end+1]
        lb  = label_np      [start:end+1]
        rtg = returns_np    [start:end+1]

        dataset_traj.append(
            Transition(
                obs         = jnp.array(o),
                action      = jnp.array(a),
                reward      = jnp.array(r),
                next_obs    = jnp.array(no),
                next_action = jnp.array(na),
                done        = jnp.array(d),
                traj_return = jnp.array(rtg),
                label       = jnp.array(lb),
            )
        )
        start = end + 1

    # 4) Append the final partial trajectory (if any)
    if start < N:
        o   = obs_np        [start:N]
        a   = action_np     [start:N]
        r   = reward_np     [start:N]
        no  = next_obs_np   [start:N]
        na  = next_action_np[start:N]
        d   = done_np       [start:N]
        lb  = label_np      [start:N]
        rtg = returns_np    [start:N]

        dataset_traj.append(
            Transition(
                obs         = jnp.array(o),
                action      = jnp.array(a),
                reward      = jnp.array(r),
                next_obs    = jnp.array(no),
                next_action = jnp.array(na),
                done        = jnp.array(d),
                traj_return = jnp.array(rtg),
                label       = jnp.array(lb),
            )
        )

    print(f"Split into {len(dataset_traj)} trajectories.")

    return dataset_traj

def split_trajectories(dataset):

    obs=onp.array(dataset["observations"])
    action=onp.array(dataset["actions"])
    reward=onp.array(dataset["rewards"])
    next_obs=onp.array(dataset["next_observations"])
    done=onp.array(dataset["terminals"])

    done_idxs  = onp.nonzero(done)[0].tolist()

    dataset_traj = []
    start = 0
    for end in done_idxs:
        # slice out Python‐numpy views
        o   = obs        [start:end+1]
        a   = action     [start:end+1]
        r   = reward     [start:end+1]
        no  = next_obs   [start:end+1]
        d   = done       [start:end+1]

        traj = {
            "observations":o,
            "actions":a,
            "rewards":r,
            "next_observations":no,
            "terminals":d,
        }
        dataset_traj.append(traj)
        start = end + 1

    print(f"Split into {len(dataset_traj)} trajectories.")
    return dataset_traj

def d4rl_dataset_preprocess(dataset):
    dataset_traj_list = []
    terms = dataset['terminals'].astype(bool)
    N = terms.shape[0]
    # build our own “timeout” mask
    timeouts = onp.zeros_like(terms)
    ctr = 0
    max_length = 1000
    for i in range(N):
        if terms[i]:
            # real termination → reset counter
            ctr = 0
        else:
            # if we’ve already done max_length steps, mark a timeout here
            if ctr >= max_length - 1:
                timeouts[i] = True
                ctr = 0
            else:
                ctr+=1
    done = terms | timeouts
    dataset['terminals'] = done
    dataset_traj_list=split_trajectories(dataset)

    return jax.tree_util.tree_map(lambda *arrays: jnp.concatenate(arrays, axis=0), *dataset_traj_list)

def make_env_and_dataset(args, num_envs):
    # --- Initialize environment and dataset ---

    # Try OGBench first
    try:
        import ogbench
        from ogbench_dataset import get_ogbench_with_mc_calculation, get_ogbench_dataset
        if args.custom_dataset is not None:
            with open(args.custom_dataset, "rb") as f:
                raw_dataset = pickle.load(f)
        else:
            raw_dataset, norm_params = get_ogbench_with_mc_calculation(args.dataset)

        def make_single_gymnasium():
            # OGBench returns Gymnasium env when env_only=True
            return ogbench.make_env_and_datasets(args.dataset, env_only=True)

        env = SyncVectorEnv([make_single_gymnasium for _ in range(num_envs)])
        env_meta = dict(kind="ogbench", make_single_unwrapped=make_single_gymnasium)
        return env, dict(
            observations=raw_dataset["observations"],
            actions=raw_dataset["actions"],
            rewards=raw_dataset["rewards"],
            next_observations=raw_dataset["next_observations"],
            dones=raw_dataset["dones"],  # already boolean/int
        ), env_meta
    except Exception as og_err:
        og_err_msg = str(og_err)

    # Fall back to D4RL
    try:
        import d4rl
        # Build vector env via Gymnasium wrapper
        def make_single_classic():
            return gym_legacy.make(args.dataset)  # single classic env

        def make_single_wrapped():
            return ClassicGymToGymnasium(make_single_classic(), render_mode=None)

        env = SyncVectorEnv([make_single_wrapped for _ in range(num_envs)])
        if args.custom_dataset is not None:
            with open(args.custom_dataset, "rb") as f:
                ds = pickle.load(f)
        else:
            ds = d4rl.qlearning_dataset(make_single_classic())
            ds = d4rl_dataset_preprocess(ds)
        raw = dict(
            observations=ds["observations"],
            actions=ds["actions"],
            rewards=ds["rewards"],
            next_observations=ds["next_observations"],
            dones=ds["terminals"],  # treat terminals as dones; timeouts handled in preprocess
        )
        if args.use_labelled_dataset:
            raw["labels"] = ds["labels"]
        env_meta = dict(kind="d4rl", make_single_unwrapped=make_single_classic)
        return env, raw, env_meta
    except Exception as d4_err:
        d4_err_msg = str(d4_err)
        
    try:
        if args.custom_dataset is not None:
            with open(args.custom_dataset, "rb") as f:
                raw_dataset = pickle.load(f)
            def make_single_classic():
                return gym.make(args.dataset)
                
            env = SyncVectorEnv([make_single_classic for _ in range(num_envs)])
            
        env_meta = dict(kind="custom", make_single_unwrapped=make_single_classic)
        return env, raw_dataset, env_meta
    except Exception as d4_err:
        d_err_msg = str(d4_err)

    raise RuntimeError(
        f"Dataset '{args.dataset}' not found in OGBench or D4RL.\n"
        f"OGBench error: {og_err_msg}\nD4RL error: {d4_err_msg}"
        f"OVERALL: {d_err_msg}"
    )

def build_transition_from_raw(raw, gamma: float) -> Transition:
    # ensure types/shapes
    obs         = jnp.asarray(raw["observations"], dtype=jnp.float32)
    actions     = jnp.asarray(raw["actions"], dtype=jnp.float32)
    rewards     = jnp.asarray(raw["rewards"], dtype=jnp.float32).reshape(-1)
    next_obs    = jnp.asarray(raw["next_observations"], dtype=jnp.float32)
    dones       = jnp.asarray(raw["dones"]).astype(jnp.uint8).reshape(-1)

    # next_action that does NOT cross episode boundaries
    pad_last    = actions[-1:]
    shifted     = jnp.concatenate([actions[1:], pad_last], axis=0)
    next_action = jnp.where(dones[:, None] == 1, actions, shifted)

    # per-step return-to-go within episodes
    dones_f = dones.astype(jnp.float32)
    def rtg_scan(rev_carry, x):
        r, d = x
        G_next = rev_carry
        G = r + gamma * (1.0 - d) * G_next
        # if episode ends here, next step's return should reset
        G = jnp.where(d > 0, r, G)
        return G, G

    _, G_rev = jax.lax.scan(rtg_scan, jnp.array(0.0, dtype=jnp.float32), (rewards[::-1], dones_f[::-1]))
    traj_return = G_rev[::-1]

    labels = jnp.zeros((obs.shape[0],), dtype=jnp.uint8)

    return Transition(
        obs=obs,
        action=actions,
        reward=rewards,
        next_obs=next_obs,
        next_action=next_action,
        done=dones,
        traj_return=traj_return,
        label=labels,
    )