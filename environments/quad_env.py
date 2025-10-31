from typing import Optional
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from dataclasses import dataclass, field
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environments.quad_dynamics import QuadrotorDynamics, QuadState, quaternion_to_rotation_matrix

@dataclass
class EnvParams:
    init_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    init_ori: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    gate_location: np.ndarray = field(default_factory=lambda: np.array([[4.0, 4.0, 2.0]]))    

class QuadrotorEnv(gym.Env):
    """
    Gymnasium-compatible quadrotor environment with gate-following reward.
    State layout: [pos(3), quat(4) (w,x,y,z), vel(3), body_rate(3)] -> 13 floats
    Action: 4 motor thrusts (floats)
    """

    def __init__(self, max_episode_steps: int = 1000):
        super().__init__()
        self.params = EnvParams()
        self.dynamics = QuadrotorDynamics()

        # copy initial values to avoid accidental shared mutation
        self.init_pos = np.asarray(self.params.init_pos, dtype=float).copy()
        self.init_ori = np.asarray(self.params.init_ori, dtype=float).copy()
        self.gate_locations = np.asarray(self.params.gate_location, dtype=float).copy()

        # state
        self.position = self.init_pos.copy()
        self.orientation = self.init_ori.copy()
        self.velocity = np.zeros(3, dtype=float)
        self.body_rate = np.zeros(3, dtype=float)
        self.motor_thrusts = np.zeros(4, dtype=float)
        
        self.quad_state = QuadState(self.position, 
                               self.orientation, 
                               self.velocity, 
                               self.body_rate)

        self.current_gate_idx = 0
        self.gate_radius = 0.5
        self.dt = float(self.dynamics.dt) if hasattr(self.dynamics, "dt") else 0.01

        # episode bookkeeping
        self.max_episode_steps = int(max_episode_steps)
        self.episode_steps = 0

        # Spaces for SB3 / wrappers
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        # set action_space to normalized -1..1 per motor
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # compute hover thrust (per motor)
        mass = float(self.dynamics.mass)
        g_vec = np.asarray(self.dynamics.gravity)
        g_mag = -g_vec[2] if g_vec[2] < 0 else g_vec[2]
        self.hover_total = mass * g_mag
        self.hover_per_motor = self.hover_total / 4.0
        # maximum delta we allow (so thrust stays in bounds)
        self.max_delta = min(self.dynamics.max_thrust - self.hover_per_motor,
                            self.hover_per_motor - self.dynamics.min_thrust)
        # clamp max_delta to a positive number
        self.max_delta = max(0.0, float(self.max_delta))

    def _get_obs(self):
        # core state for internal dynamics (13-dim)
        core = np.hstack([self.position, self.orientation, self.velocity, self.body_rate]).astype(float)
        # relative vector to current target gate (3-dim)
        rel = (self.gate_locations[self.current_gate_idx] - self.position).astype(float)
        # return augmented observation for the policy (16 dims)
        return np.hstack([core, rel]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Gymnasium-style reset; seed is optional
        if seed is not None:
            np.random.seed(seed)
        self.quad_state.position = self.init_pos.copy()
        self.quad_state.orientation = self.init_ori.copy()
        self.quad_state.velocity[:] = 0.0
        self.quad_state.body_rate[:] = 0.0
        self.motor_thrusts[:] = 0.0
        self.current_gate_idx = 0
        self.episode_steps = 0
        
        self.update_state()

        obs = self._get_obs()
        info = {}
        return obs, info

    def update_state(self):
        self.position = self.quad_state.position.copy()
        self.orientation = self.quad_state.orientation.copy()
        self.velocity = self.quad_state.velocity.copy()
        self.body_rate = self.quad_state.body_rate.copy()

    def step(self, action):
        old_pos = self.position
        # action in [-1,1]
        action = np.asarray(action, dtype=float).reshape(-1)[:4]
        # map to thrusts around hover
        motor_thrusts = self.hover_per_motor + action * self.max_delta
        # clip to actuator bounds
        motor_thrusts = np.clip(motor_thrusts, self.dynamics.min_thrust, self.dynamics.max_thrust)
        # pass motor_thrusts to dynamics
        self.quad_state = self.dynamics.run(self.quad_state, motor_thrusts, self.dt)
        self.update_state()
        
        # target location
        target = self.gate_locations[self.current_gate_idx]
        # current_distance to target
        dist = float(np.linalg.norm(self.position - target))
        old_dist = float(np.linalg.norm(old_pos - target))
        reward = -0.008 * dist + 0.2 * (old_dist - dist)
        # bookkeeping
        self.episode_steps += 1

        # reward + gate progression
        if dist < self.gate_radius:
            reward += 5.0
            print(f"PASSED GATE: {self.current_gate_idx}")
            if self.current_gate_idx < len(self.gate_locations) - 1:
                self.current_gate_idx += 1

        # termination conditions
        terminated = False
        # crash (under ground)
        if self.position[2] < 0.0:
            reward -= 10.0
            terminated = True
            

        # passed final gate
        if (self.current_gate_idx == len(self.gate_locations) - 1) and (np.linalg.norm(self.position - self.gate_locations[-1]) < self.gate_radius):
            reward += 100
            terminated = True

        # truncated due to max length
        truncated = False
        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        info = {}
        if truncated:
            info["TimeLimit.truncated"] = True
        if terminated or truncated:
            info["episode"] = {"r": float(reward), "l": int(self.episode_steps)}

        # normalize quaternion so observation is well-formed
        q_idx = slice(3, 7)
        q = self._get_obs()[q_idx]
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            self.orientation = self.orientation / q_norm

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

     