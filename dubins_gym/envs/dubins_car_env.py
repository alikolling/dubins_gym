import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Optional

class DubinsEnv5D(gym.Env):
    """Dubins car environment with 5D state: [x, y, sin(theta), cos(theta), v].
    
    def __init__(self):
        self.render_mode = None
        self.dt = 0.05
        self.max_steps = 1000
        self.current_step = 0
        self.u_max = 1.25  # Max angular rate
        self.a_max = 1.0  # Max acceleration
        self.v_max = 3.0  # Max velocity
        self.obs_high = np.array([4., 4., 1., 1., self.v_max])
        self.obs_low = np.array([-4., -4., -1., -1., -self.v_max])
        self.act_high = np.array([1., 1.])
        self.act_low = np.array([-1., -1.])
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=self.act_low, high=self.act_high, dtype=np.float32)
        self.constraint = [{"center": (0., 0.), "radius": 0.5}]  # [x, y, r] for avoidance circle
        self.viewer = None  # For rendering

        self.uncertainty_zones = [
           {"center": (-2.5, -2.5), "radius": 1.5},
           {"box": (-1, 1, 2, 3)}
        ]
        self.goal = np.array((2., 2.), dtype=np.float32)
        self.goal_radius = 0.3

    def dist_goal_func(self, position):
        return np.linalg.norm(position - self.goal)
        
    def step(self, action):
        self.current_step += 1
        # Scale actions
        omega = action[1] * self.u_max  # Angular rate
        accel = action[0] * self.a_max  # Max acceleration

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
        self.state[4] = np.clip(self.state[4], -self.v_max , self.v_max)

        # Reward: Negative for violation (inside circle), 0 otherwise
        cx, cy = self.constraint[0]['center']
        cr = self.constraint[0]['radius']
        dist_sq = (self.state[0] - cx)**2 + (self.state[1] - cy)**2
        rew = -max(0, cr**2 - dist_sq) * 10.0  # Stronger penalty  # Penalty if inside r

        terminated = False
        truncated = self.current_step >= self.max_steps
        if np.any(self.state[:2] > self.obs_high[:2]) or np.any(self.state[:2] < self.obs_low[:2]) or dist_sq < cr**2:
            terminated = True
            rew -= 50.0  # Large penalty for violation
        
        x, y = self.state[0], self.state[1]
        info = {"distance_sq": dist_sq}
        info["uncertainty_zone"] = any(
            (
                (x - zone["center"][0])**2 + (y - zone["center"][1])**2 < zone["radius"]**2
                if "center" in zone
                else zone["box"][0] < x < zone["box"][2] and zone["box"][1] < y < zone["box"][3]
                if "box" in zone
                else False
            )
            for zone in self.uncertainty_zones
)
        
        # compute distance to goal
        dist_goal = self.dist_goal_func(self.state[:2])
        #self.state[5] = dist_goal
        info['dist_to_goal'] = dist_goal

       # Dense reward
        rew += -0.5 * dist_goal
        rew -= 0.005
        if dist_goal < self.goal_radius:
            rew += 50.0
        
        return self.state.astype(np.float32), rew, terminated, truncated, info


    def _is_in_uncertainty_zone(self, x, y):
        for zone in self.uncertainty_zones:
            if 'center' in zone:
                zx, zy = zone['center']; zr = zone['radius']
                if (x - zx)**2 + (y - zy)**2 < (zr+0.1)**2:
                    return True
            elif 'box' in zone:
                xmin, ymin, xmax, ymax = zone['box']
                if xmin < x+0.1 < xmax and ymin < y+0.1 < ymax:
                    return True
        return False

    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)  # Sets self.np_random correctly as np.random.Generator
        self.current_step = 0
        
        if options and "initial_state" in options:
            self.state = np.array(options["initial_state"], dtype=np.float32)
        else:
            # Keep sampling until state is valid (outside constraint and uncertainty zones)
            while True:
                # Sample theta and velocity
                theta = self.np_random.uniform(low=0, high=2 * np.pi)
                v = self.np_random.uniform(low=-self.v_max, high=self.v_max)  # Sample velocity
                # Sample x, y within bounds
                x, y = self.np_random.uniform(low=self.obs_low[:2], high=self.obs_high[:2])
                
                # Check constraint circle: (x - c_x)^2 + (y - c_y)^2 >= r^2
                cx, cy = self.constraint[0]['center']
                cr = self.constraint[0]['radius']

                if (x - cx)**2 + (y - cy)**2 < (cr+0.1)**2:
                    continue  # Inside constraint circle, resample
                
                if self._is_in_uncertainty_zone(x, y):
                    continue
                
                # Valid state found
                self.state = np.array(
                    [x, y, np.sin(theta), np.cos(theta), v], dtype=np.float32
                )
                break
        
        # Compute dist_to_goal for state[5]
        #self.state[5] = self.dist_goal_func(self.state[:2])
        info = {"uncertainty_zones":self.uncertainty_zones, "goal":{"goal_pos":self.goal, "goal_radius":self.goal_radius}, "obstacle": self.constraint}
        return self.state.copy(), info
    
    def render(self, mode='human', trajectory=None, traj_uncertainties=None):
        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(self.obs_low[0] - 0.5, self.obs_high[0] + 0.5)
            self.ax.set_ylim(self.obs_low[1] - 0.5, self.obs_high[1] + 0.5)
            self.ax.set_aspect('equal')
            
            # Plot constraint (unsafe) region
            cx, cy = self.constraint[0]['center']
            cr = self.constraint[0]['radius']
            self.ax.add_patch(plt.Circle((cx, cy), cr, color='r', alpha=0.5, label='Unsafe'))
            
            #Plot goal region
            gx, gy = self.goal
            self.ax.add_patch(plt.Circle((gx, gy), self.goal_radius, color='g', alpha=0.3, label='Goal'))
            
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
            del self.ax"""
            

    """Dubins car environment with 5D state: [x, y, sin(theta), cos(theta), v]."""
    
    def __init__(self):
        self.render_mode = None
        self.dt = 0.05
        self.max_steps = 200  # Aligned with shorter horizon for faster TD-MPC2 training
        self.current_step = 0
        self.u_max = 1.25  # Max angular rate
        self.a_max = 0.5  # Increased acceleration for better dynamics
        self.v_max = 1.0  # Max velocity
        self.obs_high = np.array([4., 4., 1., 1., self.v_max])
        self.obs_low = np.array([-4., -4., -1., -1., -self.v_max])
        self.act_high = np.array([1., 1.])
        self.act_low = np.array([-1., -1.])
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=self.act_low, high=self.act_high, dtype=np.float32)
        self.constraint = [{"center": (0., 0.), "radius": 0.5}]  # Avoidance circle
        self.viewer = None  # For rendering

        self.uncertainty_zones = [
            {"center": (-2.5, -2.5), "radius": 1.5},
            {"box": (-1, 1, 2, 3)}
        ]
        self.goal = np.array((2., 2.), dtype=np.float32)
        self.goal_radius = 0.3
        self.beta = 0.1  # Scaling factor from Safety-Gym
        self.reward_goal = 1.0  # Goal bonus per step from Safety-Gym
        self.last_dist = 0.0  # For delta distance reward

    def dist_goal_func(self, position):
        return np.linalg.norm(position - self.goal)
        
    def step(self, action):
        self.current_step += 1
        # Scale actions
        omega = action[1] * self.u_max
        accel = action[0] * self.a_max

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
        self.state[4] = np.clip(self.state[4], -self.v_max, self.v_max)

        # Reward: Negative for violation
        cx, cy = self.constraint[0]['center']
        cr = self.constraint[0]['radius']
        dist_sq = (self.state[0] - cx)**2 + (self.state[1] - cy)**2
        rew = 0.0
        if dist_sq < cr**2:
            rew = - 1.0  # Penalty per step inside constraint, aligned with Safety-Gym cost scale

        terminated = False
        truncated = self.current_step >= self.max_steps
        if np.any(self.state[:2] > self.obs_high[:2]) or np.any(self.state[:2] < self.obs_low[:2]) or dist_sq < cr**2:
            terminated = True
            rew = -1.0  # Large penalty on termination (keep for now, but tunable)
        
        x, y = self.state[0], self.state[1]
        info = {"distance_sq": dist_sq}
        info["uncertainty_zone"] = any(
            (
                (x - zone["center"][0])**2 + (y - zone["center"][1])**2 < zone["radius"]**2
                if "center" in zone
                else zone["box"][0] < x < zone["box"][2] and zone["box"][1] < y < zone["box"][3]
                if "box" in zone
                else False
            )
            for zone in self.uncertainty_zones
        )
        
        # Compute distance to goal
        dist_goal = self.dist_goal_func(self.state[:2])
        info['dist_to_goal'] = dist_goal

        # Safety-Gym-style rewards
        rew = self.beta * (self.last_dist - dist_goal)  # Reward for getting closer
        self.last_dist = dist_goal
        if dist_goal < self.goal_radius:
            rew = self.reward_goal  # Bonus per step in goal region
        
        return self.state.astype(np.float32), rew, terminated, truncated, info

    def _is_in_uncertainty_zone(self, x, y):
        for zone in self.uncertainty_zones:
            if 'center' in zone:
                zx, zy = zone['center']; zr = zone['radius']
                if (x - zx)**2 + (y - zy)**2 < (zr+0.1)**2:
                    return True
            elif 'box' in zone:
                xmin, ymin, xmax, ymax = zone['box']
                if xmin < x+0.1 < xmax and ymin < y+0.1 < ymax:
                    return True
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        
        if options and "initial_state" in options:
            self.state = np.array(options["initial_state"], dtype=np.float32)
        else:
            while True:
                theta = self.np_random.uniform(low=0, high=2 * np.pi)
                v = self.np_random.uniform(low=0.0, high=self.v_max)  # Sample velocity
                x, y = self.np_random.uniform(low=self.obs_low[:2], high=self.obs_high[:2])
                
                cx, cy = self.constraint[0]['center']
                cr = self.constraint[0]['radius']
                if (x - cx)**2 + (y - cy)**2 < (cr+0.1)**2:
                    continue
                
                if self._is_in_uncertainty_zone(x, y):
                    continue
                
                self.state = np.array(
                    [x, y, np.sin(theta), np.cos(theta), v], dtype=np.float32
                )
                break
        
        self.last_dist = self.dist_goal_func(self.state[:2])  # Initialize for delta reward
        info = {"uncertainty_zones": self.uncertainty_zones, "goal": {"goal_pos": self.goal, "goal_radius": self.goal_radius}, "obstacle": self.constraint}
        return self.state.copy(), info
    
    def render(self, mode='human', trajectory=None, traj_uncertainties=None):
        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(self.obs_low[0] - 0.5, self.obs_high[0] + 0.5)
            self.ax.set_ylim(self.obs_low[1] - 0.5, self.obs_high[1] + 0.5)
            self.ax.set_aspect('equal')
            
            cx, cy = self.constraint[0]['center']
            cr = self.constraint[0]['radius']
            self.ax.add_patch(plt.Circle((cx, cy), cr, color='r', alpha=0.5, label='Unsafe'))
            
            gx, gy = self.goal
            self.ax.add_patch(plt.Circle((gx, gy), self.goal_radius, color='g', alpha=0.3, label='Goal'))
            
            for zone in getattr(self, 'uncertainty_zones', []):
                if 'center' in zone:
                    zx, zy = zone['center']
                    zr = zone.get('radius', 0.2)
                    self.ax.add_patch(plt.Circle((zx, zy), zr, color='orange', alpha=0.3, label='OOD zone'))
                elif 'box' in zone:
                    xmin, xmax, ymin, ymax = zone['box']
                    self.ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='orange', alpha=0.3, label='OOD zone'))

            self.car_arrow = None

        x, y, sin_theta, cos_theta, v = self.state
        theta = np.arctan2(sin_theta, cos_theta)
        arrow_length = 0.1 * abs(v) / self.v_max + 0.05
        if self.car_arrow is not None:
            self.car_arrow.remove()
        self.car_arrow = self.ax.arrow(x, y, arrow_length*np.cos(theta), arrow_length*np.sin(theta), head_width=0.05, color='b')

        if trajectory is not None:
            traj = np.array(trajectory)
            if traj_uncertainties is not None:
                sc = self.ax.scatter(traj[:,0], traj[:,1], c=traj_uncertainties, cmap='coolwarm', label='Trajectory (unc)')
                self.fig.colorbar(sc, ax=self.ax, label='Uncertainty')
            else:
                self.ax.plot(traj[:,0], traj[:,1], '-o', color='blue', label='Trajectory')

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

