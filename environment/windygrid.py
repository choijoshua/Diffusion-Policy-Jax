from typing import Optional
import numpy as np
import gymnasium as gym


def seg_intersection(p1, p2, p3, p4):
    """
    p1, p2, p3, p4: each a tuple (x, y)
    Returns (x, y) of the intersection point if the segments [p1–p2] and [p3–p4]
    intersect (including colinear overlap, returning the midpoint of the overlap),
    otherwise returns None.
    """
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1])
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    r = sub(p2, p1)
    s = sub(p4, p3)
    rxs = cross(r, s)
    qp = sub(p3, p1)
    qpxr = cross(qp, r)

    # Parallel or colinear
    if rxs == 0:
        if qpxr != 0:
            return None  # parallel non-colinear
        # colinear → project onto largest-axis for overlap check
        if abs(r[0]) > abs(r[1]):
            # avoid divide-by-zero
            t0 = (p3[0] - p1[0]) / r[0] if r[0] != 0 else 0
            t1 = (p4[0] - p1[0]) / r[0] if r[0] != 0 else 0
        else:
            t0 = (p3[1] - p1[1]) / r[1] if r[1] != 0 else 0
            t1 = (p4[1] - p1[1]) / r[1] if r[1] != 0 else 0
        t_min, t_max = min(t0, t1), max(t0, t1)
        # no overlap
        if t_max < 0 or t_min > 1:
            return None
        # return midpoint of overlap
        t_mid = max(0, t_min) + (min(1, t_max) - max(0, t_min)) / 2
        return (p1[0] + t_mid * r[0], p1[1] + t_mid * r[1])

    # Proper intersection
    t = cross(qp, s) / rxs
    u = cross(qp, r) / rxs
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p1[0] + t * r[0], p1[1] + t * r[1])

    return None

class WindyGridEnv(gym.Env):

    def __init__(self, width: int = 15, height: int = 10):
        self.width = width
        self.height = height
        self.max_timesteps = 100
        self.timesteps = 0
        self.successful = False
        self.stuck = False

        self.boundary = [(np.array([0, 0]), np.array([0, self.height])),
                         (np.array([0, self.height]), np.array([self.width, self.height])),
                         (np.array([self.width, self.height]), np.array([self.width, 0])),
                         (np.array([self.width, 0]), np.array([0, 0]))]

        self._agent_location = np.array([0, self.height / 2])
        self._target_location = np.array([self.width, self.height / 2])

        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([width, height]))

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
    
    def _get_obs(self):
        return self._agent_location
    
    def _get_info(self):
        return {"distance": self.get_distance(self._agent_location)}
    
    def get_distance(self, pos):
        return np.linalg.norm(
                pos - self._target_location, ord=1)
        
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.timesteps = 0
        self.successful = False
        self.stuck = False

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([0, self.height / 2])

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = np.array([self.width, self.height / 2])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def reached_goal(self, new_pos):
        x = new_pos[0]
        y = new_pos[1]
        goal_threshold = (self._target_location[1] - 0.5, self._target_location[1] + 0.5)

        if x == self.width and y >= goal_threshold[0] and y <= goal_threshold[1]:
            return True
        
        return False

    def process_new_pos(self, prev_pos, new_pos):
        for seg in self.boundary:
            point = seg_intersection(prev_pos, new_pos, seg[0], seg[1])
            if point is not None:
                return np.array([point[0], point[1]])
        
        print("Something Wrong, must have a point intersecting")
        return None
    
    def stochastic_wind(self, seed: Optional[int] = None):
        i = np.random.random()
        if i < 0.25:
            if self._agent_location[1] < self.height/2:
                return np.array([0, -1.5])
            else:
                return np.array([0, 1.5])
        return np.array([0, 0])
    
    def step(self, action):
        self.timesteps += 1
        wind = self.stochastic_wind()
        if self.stuck or self.successful:
            action = np.array([0, 0])
            wind = np.array([0, 0])

        new_pos = self._agent_location + action + wind
        low = np.array([0, 0])
        high = np.array([self.width, self.height])
        if np.any((new_pos < low) | (new_pos > high)):
            new_pos = self.process_new_pos(self._agent_location, new_pos)
            if self.reached_goal(new_pos):
                self.successful = True
                print("SUCCCESSFUL")
            else:
                self.stuck = True

        terminated = False
        truncated = False
        if self.timesteps >= self.max_timesteps:
            truncated = True

        if self.successful:
            reward = 10
        elif self.stuck:
            reward = -2
        else:
            reward = -0.02*self.get_distance(new_pos)

        self._agent_location = new_pos
            
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
