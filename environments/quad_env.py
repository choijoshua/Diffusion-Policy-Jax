from typing import Optional
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from dataclasses import dataclass, field

from quad_dynamics import QuadrotorDynamics, quaternion_to_rotation_matrix

@dataclass
class EnvParams:
    init_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    init_ori: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    gate_location: np.ndarray = field(default_factory=lambda: np.array([[4.0, 4.0, 1.9],
                                                                        [8.0, 5.0, 2.0],
                                                                        [11.0, -1.0, 1.5],
                                                                        [3.0, 2.0, 1.7],
                                                                        [0.0, 0.0, 2.0]]))    

class QuadrotorEnv(gym.Env):
    
    def __init__(self):
        self.params = EnvParams()
        self.dynamics = QuadrotorDynamics()
        self.position = self.params.init_pos
        self.orientation = self.params.init_ori
        self.gate_locations = self.params.gate_location
        self.velocity = np.array([0, 0, 0])
        self.body_rate = np.array([0, 0, 0])
        self.motor_thrusts = np.array([0, 0, 0, 0])
        self.current_gate_idx = 0  # index of the gate the drone is currently aiming for
        self.gate_radius = 0.5     # radius around the gate considered "passed"
        self.dt = 0.01
        
    def _get_obs(self):
        state = np.hstack([self.position, self.orientation, self.velocity, self.body_rate])
        return state
    
    def _get_position(self):
        return self.position
    
    def _get_info(self):
        return 
    
    def reset(self):
        self.position = self.params.init_pos
        self.orientation = self.params.init_ori
        self.velocity = np.array([0, 0, 0])
        self.body_rate = np.array([0, 0, 0])
        self.motor_thrusts = np.array([0, 0, 0, 0])
        self.dt = 0.01
        self.current_gate_idx = 0
        return self._get_obs()
    
    def update_state(self, state):
        
        self.position = state[:3]
        self.orientation = state[3:7]
        self.velocity = state[7:10]
        self.body_rate = state[10:]
        return
    
    def compute_reward(self):
        target_gate = self.gate_locations[self.current_gate_idx]
        dist = np.linalg.norm(self.position - target_gate)

        reward = 0.0

        # reward for getting closer (inverse of distance)
        reward += 1.0 / (dist + 1e-6)

        # bonus for passing the gate
        if dist < self.gate_radius:
            reward += 10.0
            # move to the next gate if not last
            if self.current_gate_idx < len(self.gate_locations) - 1:
                self.current_gate_idx += 1

        return reward
    
    def step(self, input):
        obs = self._get_obs()
        new_obs = self.dynamics.run(obs, input, self.dt)
        self.update_state(new_obs)
        
        target_gate = self.gate_locations[self.current_gate_idx]
        dist = np.linalg.norm(self.position - target_gate)

        reward = 0.0

        # reward for getting closer (inverse of distance)
        reward += 0.01*(-dist)

        # termination  reward
        if self.position[-1] < 0:
            reward += -10
            termination = True
            
        # bonus for passing the gate
        if dist < self.gate_radius:
            reward += 10.0
            # move to the next gate if not last
            if self.current_gate_idx < len(self.gate_locations) - 1:
                self.current_gate_idx += 1
                
        termination = self.current_gate_idx == len(self.gate_locations) - 1 \
           and np.linalg.norm(self.position - self.gate_locations[-1]) < self.gate_radius
        # gym APIs expect (obs, reward, terminated, truncated, info)
        return new_obs, reward, termination, False, {}
    
    def render(
        self,
        motor_cmd_fn=None,
        sim_time: float = 5.0,
        axes_length: float = 2.0,
        show_traj: bool = True,
        fps: int = 60,
        elev: float = 30,
        azim: float = -60,
    ):
        if motor_cmd_fn is None:
            def motor_cmd_fn(t, state, dynamics):
                mass = dynamics.mass
                g_vec = dynamics.gravity
                g_mag = -g_vec[2] if g_vec[2] < 0 else g_vec[2]
                per = (1.1*mass * g_mag) / 4.0
                return np.array([per, per, per, per])

        steps = int(np.ceil(sim_time / self.dt))
        traj_pos = np.zeros((steps, 3))
        traj_quat = np.zeros((steps, 4))

        # run simulation and record
        state = self.reset()
        for i in range(steps):
            t = i * self.dt
            u = motor_cmd_fn(t, state, self.dynamics)
            obs, _, _, _, _ = self.step(u)
            # normalize quaternion to be safe
            obs[3:7] = obs[3:7] / (np.linalg.norm(obs[3:7]) + 1e-12)
            traj_pos[i] = obs[0:3]
            traj_quat[i] = obs[3:7]
            state = obs

        # set up figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1,1,1))

        # compute plot limits from trajectory (with padding)
        all_xyz = traj_pos
        pad = max(0.5, axes_length)
        xmid = (all_xyz[:,0].min() + all_xyz[:,0].max())/2
        ymid = (all_xyz[:,1].min() + all_xyz[:,1].max())/2
        zmin = all_xyz[:,2].min()
        zmax = all_xyz[:,2].max()
        span = max(all_xyz[:,0].ptp(), all_xyz[:,1].ptp(), all_xyz[:,2].ptp(), 1.0)
        ax.set_xlim(xmid - span/2 - pad, xmid + span/2 + pad)
        ax.set_ylim(ymid - span/2 - pad, ymid + span/2 + pad)
        ax.set_zlim(max(0, zmin - pad), zmax + pad + 1.0)

        # Pre-create artists for in-place updates (fast)
        if show_traj:
            traj_line, = ax.plot([], [], [], lw=1.5, color="k", alpha=0.6)
        else:
            traj_line = None

        # center marker
        center_scatter = ax.scatter([], [], [], s=40, color="C0")

        # body-axis lines: x(red), y(green), z(blue)
        x_line, = ax.plot([], [], [], lw=3, color="r")
        y_line, = ax.plot([], [], [], lw=3, color="g")
        z_line, = ax.plot([], [], [], lw=3, color="b")

        time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

        def init():
            if traj_line is not None:
                traj_line.set_data([], [])
                traj_line.set_3d_properties([])
            center_scatter._offsets3d = ([], [], [])
            for L in (x_line, y_line, z_line):
                L.set_data([], [])
                L.set_3d_properties([])
            time_text.set_text("")
            return [traj_line, center_scatter, x_line, y_line, z_line, time_text] if traj_line is not None else [center_scatter, x_line, y_line, z_line, time_text]

        def update(frame):
            pos = traj_pos[frame]
            quat = traj_quat[frame]
            R = quaternion_to_rotation_matrix(quat)  # body -> world

            # body axes in body frame
            x_b = np.array([axes_length, 0.0, 0.0])
            y_b = np.array([0.0, axes_length, 0.0])
            z_b = np.array([0.0, 0.0, axes_length])

            x_w = R @ x_b + pos
            y_w = R @ y_b + pos
            z_w = R @ z_b + pos

            # update center
            center_scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

            # update axis lines (start at pos, end at pos + axis)
            x_line.set_data([pos[0], x_w[0]], [pos[1], x_w[1]])
            x_line.set_3d_properties([pos[2], x_w[2]])

            y_line.set_data([pos[0], y_w[0]], [pos[1], y_w[1]])
            y_line.set_3d_properties([pos[2], y_w[2]])

            z_line.set_data([pos[0], z_w[0]], [pos[1], z_w[1]])
            z_line.set_3d_properties([pos[2], z_w[2]])

            # trajectory
            if traj_line is not None:
                traj_line.set_data(traj_pos[: frame + 1, 0], traj_pos[: frame + 1, 1])
                traj_line.set_3d_properties(traj_pos[: frame + 1, 2])

            time_text.set_text(f"t = {frame*self.dt:.2f}s")
            return [traj_line, center_scatter, x_line, y_line, z_line, time_text] if traj_line is not None else [center_scatter, x_line, y_line, z_line, time_text]

        interval_ms = 1000.0 / fps
        anim = animation.FuncAnimation(fig, update, frames=steps, init_func=init, interval=interval_ms, blit=False)

        plt.show()
        return anim
    