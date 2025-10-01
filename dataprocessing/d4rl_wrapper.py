# adapter_classic_to_gymnasium.py
import gymnasium as gym
from gymnasium import spaces as gspa

# Import classic gym just for isinstance checks
import gym as gym_legacy
from gym import spaces as lspa


def _to_gymnasium_space(space):
    """Convert classic gym space -> gymnasium space (recursively)."""
    if isinstance(space, gspa.Space):
        return space  # already gymnasium

    # ---- Simple spaces ----
    if isinstance(space, lspa.Box):
        return gspa.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    if isinstance(space, lspa.Discrete):
        return gspa.Discrete(n=space.n, start=getattr(space, "start", 0))
    if isinstance(space, lspa.MultiBinary):
        return gspa.MultiBinary(n=space.n)
    if isinstance(space, lspa.MultiDiscrete):
        return gspa.MultiDiscrete(nvec=space.nvec)

    # ---- Composite spaces ----
    if isinstance(space, lspa.Tuple):
        return gspa.Tuple(tuple(_to_gymnasium_space(s) for s in space.spaces))
    if isinstance(space, lspa.Dict):
        return gspa.Dict({k: _to_gymnasium_space(s) for k, s in space.spaces.items()})

    # Fallback: try to reconstruct via attributes if it quacks like a Box
    if hasattr(space, "shape") and hasattr(space, "dtype") and hasattr(space, "low") and hasattr(space, "high"):
        return gspa.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)

    raise TypeError(f"Cannot convert legacy space of type {type(space)} to gymnasium space.")


class ClassicGymToGymnasium(gym.Env):
    """
    Wrap a classic Gym (<=0.21) env to Gymnasium API and spaces.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, legacy_env, render_mode=None):
        self.legacy_env = legacy_env
        self.render_mode = render_mode

        # Convert spaces to Gymnasium versions
        self.observation_space = _to_gymnasium_space(legacy_env.observation_space)
        self.action_space      = _to_gymnasium_space(legacy_env.action_space)

        # Forward useful attrs if present
        self.spec     = getattr(legacy_env, "spec", None)
        self.reward_range = getattr(legacy_env, "reward_range", (-float("inf"), float("inf")))
        self.metadata = getattr(legacy_env, "metadata", self.metadata)

    @property
    def unwrapped(self):
        return getattr(self.legacy_env, "unwrapped", self.legacy_env)

    def reset(self, *, seed=None, options=None):
        # Handle seeding across classic API variants
        if seed is not None:
            if hasattr(self.legacy_env, "seed"):
                try:
                    self.legacy_env.seed(seed)
                except Exception:
                    pass
            try:
                out = self.legacy_env.reset(seed=seed)
            except TypeError:
                out = self.legacy_env.reset()
        else:
            out = self.legacy_env.reset()

        # classic gym reset() often returns obs only
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.legacy_env.step(action)
        truncated_flag = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done and not truncated_flag)
        truncated  = truncated_flag
        return obs, float(reward), terminated, truncated, info

    def render(self):
        try:
            if self.render_mode is not None:
                return self.legacy_env.render(mode=self.render_mode)
            return self.legacy_env.render()
        except TypeError:
            return self.legacy_env.render()

    def close(self):
        return self.legacy_env.close()
