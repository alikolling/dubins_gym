"""Zone geometry invariants. Run directly: `python test_zones.py`.

The load-bearing one is disjointness. Cost fires on any untyped zone, so a high zone
that overlaps one makes `inside_high` imply `cost=1`. An agent avoiding that region is
then unattributable between the uncertainty signal and the cost signal, which is the
circularity objection the MUARL plan records as A7. The two typed variants shipped with
the high zone at `(-2.5,-2.5) r=1.0` nested inside the cost zone at the same centre with
`r=1.5`, so the property was violated everywhere it mattered.

Disjointness is checked two ways: analytically from the zone definitions, and by
stepping each env over a grid and asserting no state reports both.
"""

import numpy as np

from dubins_gym.envs.dubins_car_env import DubinsEnv5D
from dubins_gym.envs.dubins_car_epistemic_env import DubinsEpistemicEnv5D
from dubins_gym.envs.dubins_car_robust_env import DubinsRobustEnv5D

TYPED_ENVS = [DubinsEpistemicEnv5D, DubinsRobustEnv5D]
ALL_ENVS = TYPED_ENVS + [DubinsEnv5D]


def split(env):
    zones = env.uncertainty_zones
    return (
        [z for z in zones if z.get("type", "normal") == "normal"],
        [z for z in zones if z.get("type") == "high"],
    )


def contains(zone, x, y):
    if "center" in zone:
        zx, zy = zone["center"]
        return (x - zx) ** 2 + (y - zy) ** 2 < zone["radius"] ** 2
    xmin, ymin, xmax, ymax = zone["box"]
    return xmin < x < xmax and ymin < y < ymax


def clearance(a, b):
    """Signed gap between two zones. Positive means disjoint."""
    if "center" in a and "center" in b:
        d = np.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1])
        return d - (a["radius"] + b["radius"])
    if "box" in a and "box" in b:
        (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = a["box"], b["box"]
        return np.hypot(max(bx0 - ax1, ax0 - bx1, 0), max(by0 - ay1, ay0 - by1, 0)) or min(
            bx0 - ax1, ax0 - bx1, by0 - ay1, ay0 - by1)
    circle, box = (a, b) if "center" in a else (b, a)
    xmin, ymin, xmax, ymax = box["box"]
    cx, cy = circle["center"]
    dx = max(xmin - cx, 0, cx - xmax)
    dy = max(ymin - cy, 0, cy - ymax)
    return np.hypot(dx, dy) - circle["radius"]


def test_high_zones_are_disjoint_from_cost_zones():
    for cls in TYPED_ENVS:
        normal, high = split(cls())
        assert high, f"{cls.__name__} declares no high zone"
        for h in high:
            for n in normal:
                gap = clearance(h, n)
                assert gap > 0, f"{cls.__name__}: high {h} overlaps cost {n}, gap {gap:.3f}"


def test_disjointness_holds_when_stepping():
    for cls in TYPED_ENVS:
        env = cls()
        normal, high = split(env)
        both = [
            (x, y)
            for x in np.linspace(-4, 4, 161)
            for y in np.linspace(-4, 4, 161)
            if any(contains(z, x, y) for z in high) and any(contains(z, x, y) for z in normal)
        ]
        assert not both, f"{cls.__name__}: {len(both)} states are both high and costed, e.g. {both[0]}"


def test_high_zones_are_reachable_and_clear_of_fixtures():
    for cls in TYPED_ENVS:
        env = cls()
        _, high = split(env)
        for h in high:
            cx, cy = h["center"]
            r = h["radius"]
            assert np.all(np.abs([cx - r, cx + r, cy - r, cy + r]) <= 4.0), f"{cls.__name__}: {h} leaves the arena"
            ox, oy = env.obstacle[0]["center"]
            assert np.hypot(cx - ox, cy - oy) > r + env.obstacle[0]["radius"], f"{cls.__name__}: {h} covers the obstacle"
            gx, gy = env.goal
            assert np.hypot(cx - gx, cy - gy) > r + env.goal_radius, f"{cls.__name__}: {h} covers the goal"


def test_typed_variants_share_one_geometry():
    """Robust and epistemic differ in what a high zone does, never in where it is."""
    a, b = [cls().uncertainty_zones for cls in TYPED_ENVS]
    assert a == b, f"typed variants diverged:\n  {a}\n  {b}"


def test_only_epistemic_labels_zones():
    """`withhold_zone` keys on this, and only one env may supply it."""
    labels = {}
    for cls in ALL_ENVS:
        env = cls()
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        labels[cls.__name__] = "epistemic" in info
    assert labels["DubinsEpistemicEnv5D"], "the epistemic env must report info['epistemic']"
    assert not labels["DubinsRobustEnv5D"], "the robust env must not claim an epistemic label"
    assert not labels["DubinsEnv5D"], "the plain env must not claim an epistemic label"


def test_zone_drift_changes_dynamics_only_inside_high_zones():
    """The drift variant moves the car exactly as the plain epistemic env outside a high
    zone, and slides it sideways by `dt * drift` per step inside one."""
    import gymnasium as gym
    import dubins_gym  # noqa: F401  registers the ids

    base, drift = DubinsEpistemicEnv5D(), gym.make("dubins_gym/DubinsEpistemicDriftEnv-v0").unwrapped
    assert drift.zone_drift == 0.5 and base.zone_drift == 0.0
    action = np.array([0.3, -0.2], dtype=np.float32)
    for start, inside in (([1.5, -2.0, 0.0, 1.0, 0.6], True), ([-1.0, -3.5, 0.0, 1.0, 0.6], False)):
        for env in (base, drift):
            env.reset(options={"initial_state": start})
        _, _, _, _, ib = base.step(action)
        _, _, _, _, idr = drift.step(action)
        shift = idr["state"][:2] - ib["state"][:2]
        if inside:
            # heading 0: left of the heading is +y
            assert np.allclose(shift, [0.0, base.dt * 0.5], atol=1e-6), shift
            assert idr["epistemic"]
        else:
            assert np.allclose(shift, 0.0), shift


def test_label_marks_steps_that_start_inside():
    """A step that starts in a high zone and leaves it was moved by the zone, so it is labelled:
    withholding keyed on the end position alone stored those exits, drift included."""
    env = DubinsEpistemicEnv5D(zone_drift=0.5)
    env.reset(options={"initial_state": [2.24, -2.0, 0.0, 1.0, 1.0]})  # zone (1.5, -2) r 0.75
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert np.hypot(info["state"][0] - 1.5, info["state"][1] + 2.0) >= 0.75, "the step must leave the zone"
    assert info["epistemic"], "a step starting inside a high zone must be labelled"


def test_hazard_layout_hides_an_obstacle_in_each_high_zone():
    """`hazard` keeps the default zones and adds one obstacle at each high zone's centre."""
    import gymnasium as gym
    import dubins_gym  # noqa: F401  registers the ids

    env = gym.make("dubins_gym/DubinsEpistemicHazardEnv-v0").unwrapped
    default = DubinsEpistemicEnv5D()
    strip = lambda zs: [{k: v for k, v in z.items() if k not in ("drift", "obstacle")} for z in zs]  # noqa: E731
    assert strip(env.uncertainty_zones) == default.uncertainty_zones, "hazard must keep the default geometry"
    _, high = split(env)
    hidden = [o for o in env.obstacle if o.get("hidden")]
    assert env.obstacle[0] == default.obstacle[0] and len(hidden) == len(high) == 2
    for h, o in zip(high, hidden):
        assert o["center"] == h["center"] and o["radius"] < h["radius"]


def test_inward_drift_pulls_toward_the_zone_centre():
    import gymnasium as gym
    import dubins_gym  # noqa: F401

    env = gym.make("dubins_gym/DubinsEpistemicHazardEnv-v0").unwrapped
    env.reset(options={"initial_state": [-2.0, 0.5, 0.0, 1.0, 0.0]})  # 0.5 m east of zone A's centre, at rest
    _, _, term, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert np.allclose(info["state"][:2], [-2.0 - env.dt * 0.5, 0.5], atol=1e-6), info["state"]
    assert info["epistemic"] and not term


def test_hidden_obstacle_terminates():
    import gymnasium as gym
    import dubins_gym  # noqa: F401

    env = gym.make("dubins_gym/DubinsEpistemicHazardEnv-v0").unwrapped
    env.reset(options={"initial_state": [-2.5 + 0.32, 0.5, 0.0, -1.0, 1.0]})  # heading west at the obstacle
    _, reward, term, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert term and reward <= -10.0 + 1.0 and info["epistemic"], (term, reward)
    starts = [env.reset(seed=i)[0] for i in range(500)]
    assert not any(np.hypot(s[0] - o["center"][0], s[1] - o["center"][1]) < o["radius"]
                   for s in starts for o in env.obstacle), "no reset may start inside an obstacle"


def _frontier(zone_drift=0.5):
    return DubinsEpistemicEnv5D(zone_drift=zone_drift, layout="frontier")


def test_frontier_is_one_sided_and_clear_of_every_fixture():
    """The strip spans the arena's width up to the north wall, so data lies on one side only;
    cost zones, the goal and the visible obstacle stay at least 0.5 m below it."""
    import gymnasium as gym
    import dubins_gym  # noqa: F401

    env = gym.make("dubins_gym/DubinsFrontierEnv-v0").unwrapped
    normal, high = split(env)
    assert len(high) == 1 and high[0]["box"] == (-4.0, 1.5, 4.0, 4.0), high
    strip = high[0]
    for n in normal:
        assert clearance(strip, n) >= 0.5, f"cost zone {n} within 0.5 m of the frontier"
    gx, gy = env.goal
    assert np.allclose(env.goal, (2.5, 0.8)) and 1.5 - gy >= env.goal_radius + 0.3, env.goal
    o = env.obstacle[0]
    assert 1.5 - (o["center"][1] + o["radius"]) >= 0.5
    for n in normal:
        assert not contains(n, gx, gy) and clearance(n, {"center": (gx, gy), "radius": env.goal_radius}) > 0


def test_frontier_hides_three_obstacles_inside_the_strip():
    env = _frontier()
    hidden = [o for o in env.obstacle if o.get("hidden")]
    assert len(hidden) == 3
    for o in hidden:
        (x, y), r = o["center"], o["radius"]
        assert y - r > 1.5 and abs(x) + r < 4 and y + r < 4, o


def test_frontier_drift_pushes_north_only_inside():
    base, drift = _frontier(0.0), _frontier()
    action = np.array([0.3, -0.2], dtype=np.float32)
    for start, inside in (([1.0, 2.8, 0.0, 1.0, 0.6], True), ([1.0, 1.2, 0.0, 1.0, 0.6], False)):
        for env in (base, drift):
            env.reset(options={"initial_state": start})
        _, _, _, _, ib = base.step(action)
        _, _, _, _, idr = drift.step(action)
        shift = idr["state"][:2] - ib["state"][:2]
        expected = [0.0, base.dt * 0.5] if inside else [0.0, 0.0]
        assert np.allclose(shift, expected, atol=1e-6), (start, shift)
        assert idr["epistemic"] == inside


def test_frontier_resets_start_below_the_strip():
    env = _frontier()
    ys = [env.reset(seed=i)[0][1] for i in range(500)]
    assert max(ys) < 1.4, max(ys)
    assert max(ys) > 1.0, "starts should reach close to the frontier"



def _terrain(ice):
    import gymnasium as gym
    import dubins_gym  # noqa: F401

    return gym.make("dubins_gym/DubinsTerrainIceEnv-v0" if ice else "dubins_gym/DubinsTerrainEnv-v0").unwrapped


def test_terrain_ice_exists_only_in_the_evaluation_env():
    train, ice = _terrain(False), _terrain(True)
    assert not any(z.get("type") == "high" for z in train.uncertainty_zones) and len(train.obstacle) == 1
    assert sum(z.get("type") == "high" for z in ice.uncertainty_zones) == 2 and len(ice.obstacle) == 3
    mud = lambda env: {z["center"] for z in env.uncertainty_zones if z.get("type") == "mud"}
    assert mud(train) == mud(ice) and len(mud(train)) == 2


def test_sensor_sees_ice_before_the_car_reaches_it():
    start = [-2.5 + 1.0 + 0.8, 0.5, 0.0, -1.0, 0.0]  # 0.8 m east of zone A's edge, heading west
    for ice in (True, False):
        env = _terrain(ice)
        obs, _ = env.reset(options={"initial_state": start})
        assert obs.shape == (10,) and env.observation_space.contains(obs), obs
        assert obs[5] == 1.0, "nothing under the car yet"
        assert env.sees_ice(obs) == ice, (ice, obs[5:])


def test_mud_slows_the_car_and_reads_point_seven():
    env = _terrain(False)
    moved = {}
    for name, (x, y) in (("mud", (3.0, 2.0)), ("open", (3.0, 0.5))):
        obs, _ = env.reset(options={"initial_state": [x, y, 1.0, 0.0, 1.0]})  # heading north at full speed
        assert np.isclose(obs[5], 0.7 if name == "mud" else 1.0), obs[5:]
        nxt, *_ = env.step(np.zeros(2, dtype=np.float32))
        moved[name] = nxt[1] - y
    assert np.isclose(moved["mud"], 0.7 * moved["open"]), moved


def test_mud_is_clear_of_every_other_fixture():
    env = _terrain(True)
    for m in (z for z in env.uncertainty_zones if z.get("type") == "mud"):
        (mx, my), mr = m["center"], m["radius"]
        assert abs(mx) + mr < 4 and abs(my) + mr < 4, m
        for z in env.uncertainty_zones:
            if z is m:
                continue
            if "center" in z:
                gap = np.hypot(mx - z["center"][0], my - z["center"][1]) - z["radius"]
            else:
                xmin, ymin, xmax, ymax = z["box"]
                gap = np.hypot(mx - np.clip(mx, xmin, xmax), my - np.clip(my, ymin, ymax))
            assert gap > mr, (m, z)
        assert np.hypot(mx - env.goal[0], my - env.goal[1]) > mr + env.goal_radius
        assert all(np.hypot(mx - o["center"][0], my - o["center"][1]) > mr + o["radius"] for o in env.obstacle)



def test_batched_readings_match_stepping():
    env = _terrain(True)
    rng = np.random.default_rng(0)
    states = []
    for _ in range(200):
        theta = rng.uniform(0, 2 * np.pi)
        states.append([*rng.uniform(-3.9, 3.9, 2), np.sin(theta), np.cos(theta), rng.uniform(-1, 1)])
    states = np.array(states, dtype=np.float32)
    single = np.stack([env.reset(options={"initial_state": s})[0][5:] for s in states])
    assert np.array_equal(env.readings(states), single)
    assert (single < 0.5).any() and np.isclose(single, 0.7).any(), "the sample should see ice and mud"

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
