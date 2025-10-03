from absl import flags
from collections.abc import Mapping
from collections import namedtuple
import numpy as onp
import jax.numpy as jnp
import jax
import gym as gym_legacy
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import pickle
from dataprocessing.d4rl_wrapper import ClassicGymToGymnasium

Transition = namedtuple("Transition", "obs action reward next_obs next_action done traj_return")
FLAGS = flags.FLAGS


def split_into_trajectories(dataset):
    # 1) Pull everything out as NumPy once
    obs_np         = onp.array(dataset.obs)
    action_np      = onp.array(dataset.action)
    reward_np      = onp.array(dataset.reward)
    next_obs_np    = onp.array(dataset.next_obs)
    next_action_np = onp.array(dataset.next_action)
    done_np        = onp.array(dataset.done).astype(bool)
    rtg_np         = onp.array(dataset.traj_return)

    # 2) Compute a flat return‐to‐go array of length N
    N = reward_np.shape[0]
    done_idxs  = onp.nonzero(done_np)[0].tolist()

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
        rtg = rtg_np        [start:end+1]

        dataset_traj.append(
            Transition(
                obs         = jnp.array(o),
                action      = jnp.array(a),
                reward      = jnp.array(r),
                next_obs    = jnp.array(no),
                next_action = jnp.array(na),
                done        = jnp.array(d),
                traj_return = jnp.array(rtg)
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
        rtg = rtg_np        [start:N]

        dataset_traj.append(
            Transition(
                obs         = jnp.array(o),
                action      = jnp.array(a),
                reward      = jnp.array(r),
                next_obs    = jnp.array(no),
                next_action = jnp.array(na),
                done        = jnp.array(d),
                traj_return = jnp.array(rtg)
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
    max_length = 200
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

def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        if isinstance(batches[0][key], Mapping):
            # to concatenate batch["observations"]["image"], etc.
            concatenated[key] = concatenate_batches([batch[key] for batch in batches])
        else:
            concatenated[key] = onp.concatenate(
                [batch[key] for batch in batches], axis=0
            ).astype(onp.float32)
    return concatenated

def _determine_whether_sparse_reward(env_name):
    # return True if the environment is sparse-reward
    # determine if the env is sparse-reward or not
    if "antmaze" in env_name or env_name in [
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
        "pen-binary",
        "door-binary",
        "relocate-binary",
    ]:
        is_sparse_reward = True
    elif (
        "halfcheetah" in env_name
        or "hopper" in env_name
        or "walker" in env_name
        or "kitchen" in env_name
    ):
        is_sparse_reward = False
    elif "toy" in env_name:
        is_sparse_reward=False
    elif env_name == 'ToyDot':
        is_sparse_reward = False
    else:   
        print(env_name)
        raise NotImplementedError

    return is_sparse_reward


# used to calculate the MC return for sparse-reward tasks.
# Assumes that the environment issues two reward values: reward_pos when the
# task is completed, and reward_neg at all the other steps.
ENV_REWARD_INFO = {
    "antmaze": {  # antmaze default is 0/1 reward
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
    "adroit-binary": {  # adroit default is -1/0 reward
        "reward_pos": 0.0,
        "reward_neg": -1.0,
    },
}


def _get_negative_reward(env_name, reward_scale, reward_bias):
    """
    Given an environment with sparse rewards (aka there's only two reward values,
    the goal reward when the task is done, or the step penalty otherwise).
    Args:
        env_name: the name of the environment
        reward_scale: the reward scale
        reward_bias: the reward bias. The reward_scale and reward_bias are not applied
            here to scale the reward, but to determine the correct negative reward value.

    NOTE: this function should only be called on sparse-reward environments
    """
    if "antmaze" in env_name:
        reward_neg = (
            ENV_REWARD_INFO["antmaze"]["reward_neg"] * reward_scale + reward_bias
        )
    elif env_name in [
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
    ]:
        reward_neg = (
            ENV_REWARD_INFO["adroit-binary"]["reward_neg"] * reward_scale + reward_bias
        )
    else:
        raise NotImplementedError(
            """
            If you want to try on a sparse reward env,
            please add the reward_neg value in the ENV_REWARD_INFO dict.
        """
        )

    return reward_neg

def calc_return_to_go(
    env_name,
    rewards,
    masks,
    gamma,
    reward_scale=None,
    reward_bias=None,
    infinite_horizon=False,
):
    """
    Calculat the Monte Carlo return to go given a list of reward for a single trajectory.
    Args:
        env_name: the name of the environment
        rewards: a list of rewards
        masks: a list of done masks
        gamma: the discount factor used to discount rewards
        reward_scale, reward_bias: the reward scale and bias used to determine
            the negative reward value for sparse-reward environments. If None,
            default from FLAGS values. Leave None unless for special cases.
        infinite_horizon: whether the MDP has inifite horizion (and therefore infinite return to go)
    """
    if len(rewards) == 0:
        return onp.array([])

    # process sparse-reward envs
    if reward_scale is None or reward_bias is None:
        # scale and bias not applied, but used to determien the negative reward value
        assert reward_scale is None and reward_bias is None  # both should be unset
        reward_scale = FLAGS.reward_scale
        reward_bias = FLAGS.reward_bias
    # is_sparse_reward = _determine_whether_sparse_reward(env_name)
    is_sparse_reward = False
    if is_sparse_reward:
        reward_neg = _get_negative_reward(env_name, reward_scale, reward_bias)

    if is_sparse_reward and onp.all(onp.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        # sum up the rewards backwards as the return to go
        return_to_go = [0] * len(rewards)
        prev_return = 0 if not infinite_horizon else float(rewards[-1] / (1 - gamma))
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (
                masks[-i - 1]
            )
            prev_return = return_to_go[-i - 1]
    return onp.array(return_to_go, dtype=onp.float32)

def make_env_and_dataset(args, num_envs):
    # --- Initialize environment and dataset ---
    # Try OGBench first
    try:
        import ogbench
        from dataprocessing.ogbench_dataset import get_ogbench_with_mc_calculation
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


    return Transition(
        obs=obs,
        action=actions,
        reward=rewards,
        next_obs=next_obs,
        next_action=next_action,
        done=dones,
        traj_return=traj_return
    )