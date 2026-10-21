"""Invariants of `DubinsFrontierExploreEnv` (MUARL PLAN.md, P8). Run directly:
`python test_frontier_explore.py`."""

import gymnasium as gym
import numpy as np

import dubins_gym  # noqa: F401
from dubins_gym.envs.dubins_car_frontier_explore_env import DubinsFrontierExploreEnv
from test_zones import clearance, contains

STRIP = (-4.0, 1.5, 4.0, 4.0)
WALL = 4.0


def _env(noise=0.15 / 5 ** 0.5):
    return DubinsFrontierExploreEnv(noise=noise)


def _zones(env, kind):
    if kind == "cost":
        return [z for z in env.uncertainty_zones if "type" not in z]
    return [z for z in env.uncertainty_zones if z.get("type") == kind]


def _hidden(env, strip):
    return [z for z in _zones(env, "cost") if "center" in z and contains(strip, *z["center"])]


def test_registered_ids_differ_only_in_noise():
    hi = gym.make("dubins_gym/DubinsFrontierExploreEnv-v0").unwrapped
    lo = gym.make("dubins_gym/DubinsFrontierExploreLowNoiseEnv-v0").unwrapped
    assert np.isclose(hi.noise * 5 ** 0.5, 0.15) and np.isclose(lo.noise * 5 ** 0.5, 0.05)
    assert hi.uncertainty_zones == lo.uncertainty_zones


def test_strip_holds_the_hazards_and_the_goal():
    env = _env()
    (strip,) = _zones(env, "high")
    assert strip["box"] == STRIP
    hidden = _hidden(env, strip)
    assert len(hidden) == 3
    for z in hidden:
        (x, y), r = z["center"], z["radius"]
        assert y - r > STRIP[1] and abs(x) + r < WALL and y + r < WALL, z
    goal = {"center": tuple(env.goal), "radius": env.goal_radius}
    assert contains(strip, *env.goal), env.goal
    for z in hidden:
        assert clearance(z, goal) >= 0.5, (z, env.goal)
    assert WALL - (env.goal[1] + env.goal_radius) >= 0.5
    assert [o for o in env.obstacle if o.get("hidden")] == [], "the hazards are cost, not crashes"


def test_noise_zones_lie_south_and_clear_of_every_fixture():
    env = _env()
    (strip,) = _zones(env, "high")
    noise = _zones(env, "noise")
    assert len(noise) == 2
    fixtures = [z for z in _zones(env, "cost") if z not in _hidden(env, strip)]
    fixtures += [{"center": o["center"], "radius": o["radius"]} for o in env.obstacle]
    goal = {"center": tuple(env.goal), "radius": env.goal_radius}
    for z in noise:
        assert clearance(z, strip) >= 0.5, z
        assert clearance(z, goal) > 0, z
        for f in fixtures:
            assert clearance(z, f) >= 0.3, (z, f)
        (x, y), r = z["center"], z["radius"]
        assert abs(x) + r <= WALL - 0.3 and abs(y) + r <= WALL - 0.3, z


def test_noise_pushes_only_inside_and_the_sensor_stays_exact():
    noisy, clean = _env(), _env(noise=0.0)
    action = np.array([0.2, 0.1], dtype=np.float32)
    inside = [-2.5, 0.2, 0.0, 1.0, 0.3]
    outside = [1.0, -0.5, 0.0, 1.0, 0.3]
    shifts = []
    for i in range(400):
        for env in (noisy, clean):
            env.reset(seed=i, options={"initial_state": inside})
        obs, _, _, _, info = noisy.step(action)
        _, _, _, _, ref = clean.step(action)
        assert info["noise"] and np.array_equal(obs, info["state"])
        shifts.append(info["state"][:2] - ref["state"][:2])
    shifts = np.array(shifts)
    assert np.allclose(shifts.mean(0), 0, atol=4 * noisy.noise / 20), shifts.mean(0)
    assert np.allclose(shifts.std(0), noisy.noise, rtol=0.15), shifts.std(0)
    for env in (noisy, clean):
        env.reset(seed=0, options={"initial_state": outside})
    obs, _, _, _, info = noisy.step(action)
    _, _, _, _, ref = clean.step(action)
    assert not info["noise"] and np.array_equal(info["state"], ref["state"])


def test_hidden_hazard_costs_without_ending_the_episode():
    env = _env()
    env.reset(options={"initial_state": [0.8, 2.1, 0.0, 1.0, 0.0]})
    _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["cost"] == 1 and info["epistemic"] and not terminated


def test_resets_start_below_the_strip_and_outside_noise():
    env = _env()
    for i in range(500):
        x, y = env.reset(seed=i)[0][:2]
        assert y < 1.4 and not env.in_noise_zone(x, y), (x, y)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
