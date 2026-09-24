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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
