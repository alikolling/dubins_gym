"""The frontier with zones of process noise (MUARL PLAN.md, P8).

`LAYOUTS["frontier_explore"]`: everything north of y = 1.5 is the frontier strip (high zone,
north drift), withheld from training data. Its three hazards are cost zones rather than
obstacles, so entering them is cost the car can learn from, not a crash. The goal lies inside the
strip, so a planner trained without it must explore the strip to reach it.

Two noise zones south of the strip stay in the data. Inside one, each step pushes the car by a
zero-mean Gaussian displacement of std `noise` (m) before it moves. The observation stays exact:
the current state is in distribution and only the next one is unpredictable, which is aleatoric
uncertainty in the dynamics. Sensor noise (`DubinsRobustEnv5D`) scatters the observation itself,
which a density-based estimator reads as novelty. `info['noise']` marks a step that started
inside a noise zone. `info['epistemic']` keeps the parent's meaning.
"""

import numpy as np

from dubins_gym.envs.dubins_car_epistemic_env import DubinsEpistemicEnv5D


class DubinsFrontierExploreEnv(DubinsEpistemicEnv5D):

    def __init__(self, noise=0.15 / np.sqrt(5), zone_drift=0.5):
        """`noise`: std of the push per env step, m. The default is 0.15 m per decision of five
        steps (MUARL's action repeat), about one decision's travel at 0.6 m/s."""
        super().__init__(zone_drift=zone_drift, layout="frontier_explore")
        self.noise = float(noise)
        self.noise_zones = [z for z in self.uncertainty_zones if z.get("type") == "noise"]

    def in_noise_zone(self, x, y):
        return any((x - z["center"][0]) ** 2 + (y - z["center"][1]) ** 2 < z["radius"] ** 2
                   for z in self.noise_zones)

    def step(self, action):
        inside = self.in_noise_zone(self.state[0], self.state[1])
        if inside and self.noise:
            self.state[:2] += self.np_random.normal(0.0, self.noise, 2).astype(self.state.dtype)
        obs, reward, terminated, truncated, info = super().step(action)
        info["noise"] = inside
        return obs, reward, terminated, truncated, info
