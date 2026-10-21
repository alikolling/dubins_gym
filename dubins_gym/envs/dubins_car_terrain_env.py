"""Dubins car with a terrain sensor (MUARL PLAN.md, T1 terrain).

The observation appends five friction readings to `[x, y, sin(theta), cos(theta), v]`: under the
car, 0.5 m ahead, and 1.0 m ahead at -30, 0 and +30 degrees. Normal ground reads 1.0. Mud reads
0.7 and lets 70% of the commanded speed through; it is in both versions, so the readings carry
information in training. Ice reads 0.2 and exists only with `ice=True`: the hazard layout's high
zones, with their inward drift and hidden obstacle. Train with `ice=False` and evaluate with
`ice=True`, and the first ice reading is an input the model has never seen, which arrives up to
1 m before the car does. `info['epistemic']` keeps the parent's meaning, a step that starts or
ends on ice; `sees_ice(obs)` marks an observation whose sensor reads ice.
"""

import numpy as np
from gymnasium import spaces

from dubins_gym.envs.dubins_car_epistemic_env import DubinsEpistemicEnv5D

NORMAL = 1.0
FRICTION = {"mud": 0.7, "high": 0.2}  # by zone type; the high zones are the ice
PROBES = ((0.0, 0.0), (0.5, 0.0), (1.0, -np.pi / 6), (1.0, 0.0), (1.0, np.pi / 6))  # (m, rad), car frame


class DubinsTerrainEnv(DubinsEpistemicEnv5D):

    def __init__(self, ice=False, zone_drift=0.5):
        super().__init__(zone_drift=zone_drift, layout="terrain" if ice else "terrain_train")
        self.ice = ice
        k = len(PROBES)
        self.observation_space = spaces.Box(low=np.r_[self.obs_low, np.zeros(k)].astype(np.float32),
                                            high=np.r_[self.obs_high, np.ones(k)].astype(np.float32),
                                            dtype=np.float32)

    def friction_at(self, x, y):
        """Friction at `(x, y)`; arrays broadcast. Float64, so it compares equal to `FRICTION`."""
        out = np.full(np.broadcast(x, y).shape, NORMAL)
        for z in self.uncertainty_zones:
            if z.get("type") in FRICTION:
                inside = (x - z["center"][0]) ** 2 + (y - z["center"][1]) ** 2 < z["radius"] ** 2
                out = np.where(inside, FRICTION[z["type"]], out)
        return out

    def readings(self, states):
        """The sensor's readings for a state `[5]` or a batch `[..., 5]`."""
        s = np.asarray(states, dtype=np.float64)
        heading = np.arctan2(s[..., 2], s[..., 3])
        return np.stack([self.friction_at(s[..., 0] + d * np.cos(heading + b), s[..., 1] + d * np.sin(heading + b))
                         for d, b in PROBES], -1).astype(np.float32)

    def _speed_factor(self, x, y):
        # Ice does not slow the car; its effect is the parent's drift.
        return FRICTION["mud"] if self.friction_at(x, y) == FRICTION["mud"] else 1.0

    def _observe(self):
        return np.r_[self.state, self.readings(self.state)].astype(np.float32)

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        return self._observe(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        return self._observe(), info

    @staticmethod
    def sees_ice(obs):
        return bool((np.asarray(obs)[..., 5:] < (FRICTION["high"] + FRICTION["mud"]) / 2).any())
