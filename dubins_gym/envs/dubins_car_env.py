import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Optional

class DubinsEnv5D(gym.Env):
    """Dubins car environment with 5D state: [x, y, sin(theta), cos(theta), v]."""
    
    def __init__(self):
        self.render_mode = None
        self.dt = 0.05
        self.u_max = 1.25  # Max angular rate
        self.a_max = 0.1  # Max acceleration
        self.v_max = 1.0  # Max velocity
        self.high = np.array([4., 4., 2*np.pi, 2*np.pi, self.v_max])
        self.low = np.array([-4., -4., 0., 0., 0.0])
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.constraint = [0., 0., 0.5]  # [x, y, r] for avoidance circle
        self.viewer = None  # For rendering

        self.uncertainty_zones = [
           {"center": (0.5, 0.5), "radius": 0.3},
           {"box": (-1, 1, 2, 3)}
        ]

    def step(self, action):
        # Scale actions
        omega = action * self.u_max  # Angular rate
        accel = 0.1 * self.a_max  # Max acceleration

        # Update position
        self.state[0] += self.dt * self.state[3] * self.state[4]  # cos(theta) * v
        self.state[1] += self.dt * self.state[2] * self.state[4]  # sin(theta) * v
        
        # Update heading
        theta = np.arctan2(self.state[2], self.state[3])
        theta_next = theta + self.dt * omega
        self.state[2] = np.sin(theta_next)
        self.state[3] = np.cos(theta_next)

        # Update velocity
        self.state[4] += self.dt * accel
        self.state[4] = np.clip(self.state[4], 0, self.v_max)

        # Reward: Negative for violation (inside circle), 0 otherwise
        dist_sq = (self.state[0] - self.constraint[0])**2 + (self.state[1] - self.constraint[1])**2
        rew = -max(0, self.constraint[2]**2 - dist_sq)  # Penalty if inside r

        terminated = False
        truncated = False
        if np.any(self.state[:2] > self.high[:2]) or np.any(self.state[:2] < self.low[:2]) or dist_sq < self.constraint[2]**2:
            terminated = True
            rew -= 10.0  # Large penalty for violation
        
        x, y = self.state[0], self.state[1]
        info = {"distance_sq": dist_sq}
        info["uncertainty_zone"] = any(
            (x - zone["center"][0])**2 + (y - zone["center"][1])**2 < zone["radius"]**2
            for zone in self.uncertainty_zones if "center" in zone
        )
        return self.state.astype(np.float32), rew, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random
        
        if options:
            initial_state = options["initial_state"] if "initial_state" in options else None
        else:
            initial_state = None
        
        if initial_state is None:
            theta = self.np_random.uniform(low=0, high=2*np.pi)
            self.state = self.np_random.uniform(low=self.low, high=self.high)
            self.state[2] = np.sin(theta)
            self.state[3] = np.cos(theta)
        else:
            self.state = initial_state
        
        self.state = self.state.astype(np.float32)
        return self.state, {}
    
    def render(self, mode='human', trajectory=None, traj_uncertainties=None):
        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(self.low[0] - 0.5, self.high[0] + 0.5)
            self.ax.set_ylim(self.low[1] - 0.5, self.high[1] + 0.5)
            self.ax.set_aspect('equal')
            # Plot constraint (unsafe) region
            cx, cy, cr = self.constraint
            self.ax.add_patch(plt.Circle((cx, cy), cr, color='r', alpha=0.5, label='Unsafe'))
            # Plot goal region
            #gx, gy = self.goal
            #self.ax.add_patch(plt.Circle((gx, gy), self.goal_radius, color='g', alpha=0.3, label='Goal'))
            # Plot OOD zones
            for zone in getattr(self, 'uncertainty_zones', []):
                if 'center' in zone:
                    zx, zy = zone['center']
                    zr = zone.get('radius', 0.2)
                    self.ax.add_patch(plt.Circle((zx, zy), zr, color='orange', alpha=0.3, label='OOD zone'))
                elif 'box' in zone:
                    xmin, xmax, ymin, ymax = zone['box']
                    self.ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='orange', alpha=0.3, label='OOD zone'))

            self.car_arrow = None

        # draw car
        x, y, sin_theta, cos_theta, v = self.state
        theta = np.arctan2(sin_theta, cos_theta)
        arrow_length = 0.1 * v / self.v_max + 0.05
        if self.car_arrow is not None:
            self.car_arrow.remove()
        self.car_arrow = self.ax.arrow(x, y, arrow_length*np.cos(theta), arrow_length*np.sin(theta), head_width=0.05, color='b')
        #self.ax.plot(x, y, 'bo')

        # Optionally overlay trajectory if provided
        if trajectory is not None:
            traj = np.array(trajectory)
            if traj_uncertainties is not None:
                sc = self.ax.scatter(traj[:,0], traj[:,1], c=traj_uncertainties, cmap='coolwarm', label='Trajectory (unc)')
                self.fig.colorbar(sc, ax=self.ax, label='Uncertainty')
            else:
                self.ax.plot(traj[:,0], traj[:,1], '-o', color='blue', label='Trajectory')

        # add legend and pause/update
        self.ax.legend(loc='upper right')
        if mode != 'rgb_array':
            plt.pause(0.01)
        else:
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            del self.fig
            del self.ax
            
            

