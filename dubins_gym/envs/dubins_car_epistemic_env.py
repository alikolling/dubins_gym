import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Optional

# Cost zones are untyped. A high zone's optional `drift` is "inward" (toward the zone's centre)
# or a fixed unit vector (dx, dy); without it the car slides to the left of its heading.
# `obstacle: r` puts an obstacle of radius r at the zone's centre, `obstacles: ((centre, r), ...)`
# puts them anywhere.
LAYOUTS = {
    # Interior high zones away from every fixture: entering one changes the dynamics and
    # nothing else (MUARL PLAN.md, T1).
    "default": [
        {"center": (-2.5, -2.5), "radius": 1.5},
        {"box": (-1, 1, 2, 3)},
        {"center": (-2.5, 0.5), "radius": 1.0, "type": "high"},
        {"center": (1.5, -2.0), "radius": 0.75, "type": "high"},
    ],
    # The same zones, each hiding an obstacle its drift pulls the car into. With the zones
    # withheld, a model never sees the obstacle or the pull, so only epistemic uncertainty can
    # keep a planner out (MUARL PLAN.md, P5).
    "hazard": [
        {"center": (-2.5, -2.5), "radius": 1.5},
        {"box": (-1, 1, 2, 3)},
        {"center": (-2.5, 0.5), "radius": 1.0, "type": "high", "drift": "inward", "obstacle": 0.3},
        {"center": (1.5, -2.0), "radius": 0.75, "type": "high", "drift": "inward", "obstacle": 0.3},
    ],
    # The hazard zones as ice, plus two mud patches the car sees in training too
    # (`DubinsTerrainEnv`). `terrain_train` is the same arena before the ice.
    "terrain": [
        {"center": (-2.5, -2.5), "radius": 1.5},
        {"box": (-1, 1, 2, 3)},
        {"center": (-2.5, 0.5), "radius": 1.0, "type": "high", "drift": "inward", "obstacle": 0.3},
        {"center": (1.5, -2.0), "radius": 0.75, "type": "high", "drift": "inward", "obstacle": 0.3},
        {"center": (3.0, 2.0), "radius": 0.6, "type": "mud"},
        {"center": (-2.0, 2.8), "radius": 0.6, "type": "mud"},
    ],
    "terrain_train": [
        {"center": (-2.5, -2.5), "radius": 1.5},
        {"box": (-1, 1, 2, 3)},
        {"center": (3.0, 2.0), "radius": 0.6, "type": "mud"},
        {"center": (-2.0, 2.8), "radius": 0.6, "type": "mud"},
    ],
    # A data frontier (MUARL PLAN.md, P7): the whole arena north of y = 1.5 is one high zone, so
    # withheld data lies on one side of it only and a model has to extrapolate to imagine the car
    # crossing. Interior zones have data all round and get interpolated. The strip pulls the car
    # north and hides three obstacles; the cost box moves south, out of it.
    "frontier": [
        {"center": (-2.5, -2.5), "radius": 1.5},
        {"box": (0.5, -3.0, 2.0, -1.5)},
        {"box": (-4.0, 1.5, 4.0, 4.0), "type": "high", "drift": (0.0, 1.0),
         "obstacles": (((-2.0, 2.2), 0.3), ((0.8, 2.1), 0.3), ((2.8, 2.3), 0.3))},
    ],
    # The frontier for exploration (`DubinsFrontierExploreEnv`, MUARL PLAN.md, P8). The strip's
    # hazards are cost zones at the old obstacles' centres, so entering one costs instead of
    # crashing; they overlap the high zone on purpose and are hidden only by the withholding.
    # Noise zones push the car at random and stay in the data.
    "frontier_explore": [
        {"center": (-2.5, -2.5), "radius": 1.5},
        {"box": (0.5, -3.0, 2.0, -1.5)},
        {"box": (-4.0, 1.5, 4.0, 4.0), "type": "high", "drift": (0.0, 1.0)},
        {"center": (-2.0, 2.2), "radius": 0.5},
        {"center": (0.8, 2.1), "radius": 0.5},
        {"center": (2.8, 2.3), "radius": 0.5},
        {"center": (-2.5, 0.2), "radius": 0.8, "type": "noise"},
        {"center": (3.0, -2.0), "radius": 0.6, "type": "noise"},
    ],
}
GOALS = {
    "frontier": (2.5, 0.8),  # 0.7 m below the frontier, so plans brush it
    "frontier_explore": (-0.6, 3.0),  # inside the strip: reaching it needs strip data
}


def hidden_obstacles(zone):
    """`(centre, radius)` of each obstacle a high zone hides."""
    if "obstacle" in zone:
        return ((zone["center"], zone["obstacle"]),)
    return tuple(zone.get("obstacles", ()))


class DubinsEpistemicEnv5D(gym.Env):

    """Dubins car environment with 5D state: [x, y, sin(theta), cos(theta), v]."""
    
    def __init__(self, zone_drift=0.0, layout="default"):
        """`zone_drift` (m/s): inside a high zone the car also slides at this speed, along the
        zone's `drift` direction or else to the left of its heading. Deterministic, so
        withholding those transitions hides physics the model has never seen. At 0 a high zone
        moves the car exactly as anywhere else, so withholding it leaves a gap a model fills by
        interpolation and there is nothing for an epistemic estimator to detect (MUARL
        PLAN.md, T1). `layout` picks the zones from `LAYOUTS`."""
        self.zone_drift = float(zone_drift)
        self.layout = layout
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
        self.viewer = None  # For rendering

        # Cost fires on any untyped zone, so a high zone overlapping one would make
        # `inside_high` imply `cost=1` and leave "the agent avoided the region"
        # unattributable between uncertainty and cost. The high zones are therefore
        # disjoint from the cost zones, which `test_zones.py` asserts.
        self.uncertainty_zones = [dict(z) for z in LAYOUTS[layout]]
        # obstacle[0] is the visible one every layout shares; the rest hide inside high zones.
        self.obstacle = [{"center": (0., 0.), "radius": 0.5}] + [
            {"center": c, "radius": r, "hidden": True}
            for z in self.uncertainty_zones for c, r in hidden_obstacles(z)]
        self.goal = np.array(GOALS.get(layout, (3., -1.)), dtype=np.float32)
        self.goal_radius = 0.3
        self.beta = 1.0  # Scaling factor from Safety-Gym
        self.reward_goal = 1.0  # Goal bonus per step from Safety-Gym
        self.last_dist = 0.0  # For delta distance reward

    def dist_goal_func(self, position):
        return np.linalg.norm(position - self.goal)
        
    def step(self, action):
        self.current_step += 1
        # Scale actions
        omega = action[1] * self.u_max
        accel = action[0] * self.a_max

        # Zone membership before the move: the dynamics depend on where the car is.
        zone = self._high_zone_at(self.state[0], self.state[1])
        speed = self._speed_factor(self.state[0], self.state[1])
        if zone is not None and self.zone_drift:
            if zone.get("drift") == "inward":
                dx, dy = zone["center"][0] - self.state[0], zone["center"][1] - self.state[1]
                norm = max(np.hypot(dx, dy), 1e-6)
                dx, dy = dx / norm, dy / norm
            elif "drift" in zone:
                dx, dy = zone["drift"]
            else:  # left of the heading
                dx, dy = -self.state[2], self.state[3]
            self.state[0] += self.dt * self.zone_drift * dx
            self.state[1] += self.dt * self.zone_drift * dy

        # Update position
        self.state[0] += self.dt * self.state[3] * self.state[4] * speed  # cos(theta) * v
        self.state[1] += self.dt * self.state[2] * self.state[4] * speed  # sin(theta) * v
        
        # Update heading
        theta = np.arctan2(self.state[2], self.state[3])
        theta_next = theta + self.dt * omega
        self.state[2] = np.sin(theta_next)
        self.state[3] = np.cos(theta_next)

        # Update velocity
        self.state[4] += self.dt * accel
        self.state[4] = np.clip(self.state[4], -self.v_max, self.v_max)
        obs = self.state.copy()
        # Reward: Negative for violation
        cx, cy = self.obstacle[0]['center']
        dist_sq = (self.state[0] - cx)**2 + (self.state[1] - cy)**2
        rew, cost = 0.0, 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps
        if any((self.state[0] - o["center"][0])**2 + (self.state[1] - o["center"][1])**2 < o["radius"]**2
               for o in self.obstacle):
            terminated = True
            rew = -10.0  # Large penalty on termination (keep for now, but tunable)
        
        if np.any(self.state[:2] > self.obs_high[:2]) or np.any(self.state[:2] < self.obs_low[:2]):
            terminated = True
            rew = -10.0  # Large penalty on termination (keep for now, but tunable
        
        x, y = self.state[0], self.state[1]
        info = {"distance_sq": dist_sq}
        info["state"] = self.state.copy()
        # Separate normal and high uncertainty zones
        normal_zones = [z for z in self.uncertainty_zones if z.get("type", "normal") == "normal"]
        high_zones = [z for z in self.uncertainty_zones if z.get("type") == "high"]

        inside_normal = any(
            (
                (x - zone["center"][0])**2 + (y - zone["center"][1])**2 < zone["radius"]**2
                if "center" in zone
                else zone["box"][0] < x < zone["box"][2] and zone["box"][1] < y < zone["box"][3]
                if "box" in zone
                else False
            )
            for zone in normal_zones
        )
        
        inside_high = any(
            (
                (x - zone["center"][0])**2 + (y - zone["center"][1])**2 < zone["radius"]**2
                if "center" in zone
                else zone["box"][0] < x < zone["box"][2] and zone["box"][1] < y < zone["box"][3]
                if "box" in zone
                else False
            )
            for zone in high_zones
        )
        
        
        if inside_normal:
            cost = 1
        # A step touches a high zone if it starts or ends in one: one that starts inside drifted.
        info['epistemic'] = inside_high or zone is not None
        # Compute distance to goal
        dist_goal = self.dist_goal_func(self.state[:2])
        info['dist_to_goal'] = dist_goal

        info['cost'] = cost
        # Safety-Gym-style rewards
        rew += self.beta * (self.last_dist - dist_goal)  # Reward for getting closer
        self.last_dist = dist_goal
        if dist_goal < self.goal_radius:
            rew += self.reward_goal  # Bonus per step in goal region
        
        return obs.astype(np.float32), rew, terminated, truncated, info

    def _speed_factor(self, x, y):
        """Share of the commanded speed the ground at `(x, y)` lets through; 1 here."""
        return 1.0

    def _high_zone_at(self, x, y):
        """The high zone containing `(x, y)`, or None."""
        for zone in self.uncertainty_zones:
            if zone.get("type") != "high":
                continue
            if "center" in zone:
                if (x - zone["center"][0]) ** 2 + (y - zone["center"][1]) ** 2 < zone["radius"] ** 2:
                    return zone
            elif zone["box"][0] < x < zone["box"][2] and zone["box"][1] < y < zone["box"][3]:
                return zone
        return None

    def _is_in_uncertainty_zone(self, x, y):
        for zone in self.uncertainty_zones:
            if 'center' in zone:
                zx, zy = zone['center']; zr = zone['radius']
                if (x - zx)**2 + (y - zy)**2 < (zr+0.1)**2:
                    return True
            elif 'box' in zone:
                xmin, ymin, xmax, ymax = zone['box']
                if xmin - 0.1 < x < xmax + 0.1 and ymin - 0.1 < y < ymax + 0.1:
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
                v = self.np_random.uniform(low=-self.v_max, high=self.v_max)  # Sample velocity
                x, y = self.np_random.uniform(low=self.obs_low[:2], high=self.obs_high[:2])
                
                if any((x - o["center"][0])**2 + (y - o["center"][1])**2 < (o["radius"] + 0.1)**2
                       for o in self.obstacle):
                    continue
                
                if self._is_in_uncertainty_zone(x, y):
                    continue
                
                self.state = np.array(
                    [x, y, np.sin(theta), np.cos(theta), v], dtype=np.float32
                )
                break
        
        self.last_dist = self.dist_goal_func(self.state[:2])  # Initialize for delta reward
        info = {"uncertainty_zones": self.uncertainty_zones, "goal": {"goal_pos": self.goal, "goal_radius": self.goal_radius}, "obstacle": self.obstacle, "state": self.state.copy()}
        obs = self.state.copy()
        return obs, info
    
    def render(self, mode='human', trajectory=None, traj_uncertainties=None):
        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(self.obs_low[0] - 0.5, self.obs_high[0] + 0.5)
            self.ax.set_ylim(self.obs_low[1] - 0.5, self.obs_high[1] + 0.5)
            self.ax.set_aspect('equal')
            
            cx, cy = self.obstacle[0]['center']
            cr = self.obstacle[0]['radius']
            self.ax.add_patch(plt.Circle((cx, cy), cr, color='r', alpha=0.5, label='Obstacle'))
            
            gx, gy = self.goal
            self.ax.add_patch(plt.Circle((gx, gy), self.goal_radius, color='g', alpha=0.3, label='Goal'))
            
            for zone in getattr(self, 'uncertainty_zones', []):
                if 'center' in zone:
                    if 'type' in zone:
                        zx, zy = zone['center']
                        zr = zone.get('radius', 0.2)
                        self.ax.add_patch(plt.Circle((zx, zy), zr, color='magenta', alpha=0.3, label='High uncertainty region'))
                    else:
                        zx, zy = zone['center']
                        zr = zone.get('radius', 0.2)
                        self.ax.add_patch(plt.Circle((zx, zy), zr, color='orange', alpha=0.3, label='Unsafe region'))
                elif 'box' in zone:
                    xmin, xmax, ymin, ymax = zone['box']
                    self.ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='orange', alpha=0.3, label='Unsafe region'))

            self.car_arrow = None

        x, y, sin_theta, cos_theta, v = self.state
        theta = np.arctan2(sin_theta, cos_theta)
        arrow_length = 0.1 * abs(v) / self.v_max + 0.05
        if self.car_arrow is not None:
            self.car_arrow.remove()
        self.car_arrow = self.ax.arrow(x, y, arrow_length*np.cos(theta), arrow_length*np.sin(theta), head_width=0.1, color='b')

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
