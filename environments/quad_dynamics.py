import numpy as np
from dataclasses import dataclass, field

def quaternion_multiplication(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])

def quaternion_to_rotation_matrix(q):
    # Normalize to ensure unit quaternion
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return R

@dataclass
class QuadParams:
    motor_layout: str = "BETAFLIGHT"
    mass: float = 1.0
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    min_thrust: float = 0.0
    max_thrust: float = 7.5
    max_vel: float = 10
    max_omega: float = 15
    dt: float = 0.01
    torque_coefficient: float = 0.022
    front_motor_position: np.ndarray = field(default_factory=lambda: np.array([0.075, 0.1]))
    back_motor_position: np.ndarray = field(default_factory=lambda: np.array([0.075, 0.1]))
    J: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.001)

class QuadrotorDynamics():
    
    def __init__(self):
        self.params = QuadParams()
        self.motor_layout = self.params.motor_layout
        self.mass = self.params.mass
        self.gravity = self.params.gravity
        self.min_thrust = self.params.min_thrust
        self.max_thrust = self.params.max_thrust
        self.max_vel = self.params.max_vel
        self.max_omega = self.params.max_omega
        self.dt = self.params.dt
        self.torque_coefficient = self.params.torque_coefficient
        self.front_motor_position = self.params.front_motor_position
        self.back_motor_position = self.params.back_motor_position
        self.J = self.params.J
    
    def dynamics(self, state, motor_thrust):
        orientation = state[3:7]
        velocity = state[7:10]
        body_rate = state[10:]
        total_thrust = np.sum(motor_thrust)
        
        force = quaternion_to_rotation_matrix(orientation) @ np.array([0, 0, total_thrust])
        torque = self.get_allocation_matrix()[1:, :] @ motor_thrust
        
        position_dot = velocity
        orientation_dot = 1/2 * quaternion_multiplication(orientation, np.hstack([0, body_rate]))
        velocity_dot = 1/self.params.mass * force + self.params.gravity
        body_rate_dot = np.linalg.solve(self.J,
                                        torque - np.cross(body_rate, self.J @ body_rate))
        return np.hstack([position_dot,
                          orientation_dot,
                          velocity_dot,
                          body_rate_dot])
        
    def get_allocation_matrix(self):
        """May be overriden to specify a different motor layout"""
        front_motor_x = self.params.front_motor_position[0]
        front_motor_y = self.params.front_motor_position[1]

        back_motor_x = self.params.back_motor_position[0]
        back_motor_y = self.params.back_motor_position[1]
        torque_coeff = self.params.torque_coefficient

        if self.motor_layout == "BETAFLIGHT":
            # Compute motor torques
            return np.vstack([
                np.ones((1, 4)),  # Collective thrust contribution
                np.hstack([-back_motor_y, -front_motor_y, back_motor_y, front_motor_y]),
                np.hstack([back_motor_x, -front_motor_x, back_motor_x, -front_motor_x]),
                np.hstack([torque_coeff, -torque_coeff, -torque_coeff, torque_coeff])]
            )
        elif self.motor_layout == "PX4":
            return np.vstack([
                np.ones((1, 4)),  # Collective thrust contribution
                np.hstack([-back_motor_y, front_motor_y, -back_motor_y, front_motor_y]),
                np.hstack([-back_motor_x, front_motor_x, back_motor_x, -front_motor_x]),
                np.hstack([-torque_coeff, -torque_coeff, torque_coeff, torque_coeff])]
            )
        else:
            raise ValueError("Motor layout not supported")
        
    def run(self, state, motor_thrust, horizon):
        
        # defensive copies / types
        state = np.asarray(state, dtype=float)
        u = np.asarray(motor_thrust, dtype=float)
        u = np.clip(u, self.min_thrust, self.max_thrust)

        remaining = float(horizon)
        max_dt = float(self.dt)
        eps = 1e-12

        # step until remaining time exhausted
        while remaining > eps:
            dt = max_dt if remaining >= max_dt else remaining
            state = self.rk4_integrate_step(state, u, dt)
            remaining -= dt

        return state

    def rk4_integrate_step(self, state, motor_thrust, dt):

        state = np.asarray(state, dtype=float)
        u = np.asarray(motor_thrust, dtype=float)

        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * dt * k1, u)
        k3 = self.dynamics(state + 0.5 * dt * k2, u)
        k4 = self.dynamics(state + dt * k3, u)

        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Renormalize quaternion (indices 3:7)
        q_idx = slice(3, 7)
        q = new_state[q_idx]
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-15:
            new_state[q_idx] = q / q_norm
        else:
            # fallback: keep previous normalized quaternion
            prev_q = state[q_idx]
            prev_norm = np.linalg.norm(prev_q)
            if prev_norm > 1e-15:
                new_state[q_idx] = prev_q / prev_norm
            else:
                new_state[q_idx] = np.array([1.0, 0.0, 0.0, 0.0])

        return new_state

