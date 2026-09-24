# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

`dubins_gym` is a Gymnasium package holding **three variants of one 2D Dubins-car navigation environment**, written as the controlled testbed for the MUARL uncertainty-aware planning research. It is small enough to read end to end: one registration file and three env files of roughly 200 lines each.

`README.md` is a single heading with no content, and there are no tests, no CI, and no lint configuration. The source is the only specification.

The whole point of the package is that the three variants are **identical except in how they treat their "uncertainty zones"**. Understanding that axis is understanding the package, so it is described in full below.

## Running things

```bash
pip install -e .
```

```python
import dubins_gym          # importing the package performs registration
import gymnasium as gym

env = gym.make("dubins_gym/DubinsEnv-v0")          # or DubinsRobustEnv-v0, DubinsEpistemicEnv-v0
```

Registration is a side effect of importing `dubins_gym/__init__.py`, so the package must be imported before `gym.make`, even though the module name never appears again. Linters will flag that import as unused. Keep it.

`pyproject.toml` declares `dependencies = []`, which is wrong. The package needs `gymnasium`, `numpy`, and `matplotlib`, and `matplotlib` is imported at module load in all three env files, so it is required even for headless training that never renders. It also pins `requires-python = "==3.9.*"`, matching MUARL.

`test_zones.py` is the only test. Run it from the repo root with `python test_zones.py`; it pins the zone-geometry invariants described under "The environment", including the disjointness property that A7 violated. There is no framework and no CI. For anything it does not cover, verify a change by stepping an env directly rather than reasoning about the dynamics on paper.

## The environment

**State is 5D: `[x, y, sin(theta), cos(theta), v]`.** Heading is carried as a sine/cosine pair rather than an angle, so the observation has no wraparound discontinuity for a learned dynamics model. `step` recovers the angle with `arctan2(state[2], state[3])`, integrates it, and writes the pair back. Anything touching heading must preserve that encoding.

**Action is 2D and the ordering is easy to get backwards**: `action[0]` is acceleration (scaled by `a_max = 0.5`) and `action[1]` is angular rate (scaled by `u_max = 1.25`). Acceleration comes first.

Integration is semi-implicit and order-dependent. Position advances using the *previous* heading and velocity, then heading updates, then velocity updates and clips to `±v_max`. Reordering those three blocks changes the dynamics.

**`step` returns a five-tuple, `(obs, reward, terminated, truncated, info)`, with cost in `info['cost']`.** This is the plain Gymnasium API, and it differs from Safety-Gymnasium's six-tuple even though both feed the same MUARL trainers. MUARL's `DefaultDictWrapper` in `muarl/envs/dubins_car_env.py` is what reconciles them, lifting `info['cost']` out and returning `(obs, reward, cost, done, info)` for `TensorWrapperWithCost`.

Reward follows the Safety-Gymnasium shaping convention: a dense `beta * (last_dist - dist_goal)` term each step, plus `reward_goal` while inside the goal radius, plus a `-10.0` penalty on termination. Termination fires on entering the central obstacle or leaving the `±4` box. Truncation fires at 200 steps, which is also the registered `max_episode_steps`, so the limit is enforced twice.

`reset` rejects spawn positions inside the obstacle or inside any uncertainty zone, retrying until it finds a clear one. Note that `_is_in_uncertainty_zone` inflates each zone by `0.1` while `step` tests the exact bounds, so the agent can never start in a thin margin where it would not actually be penalized.

### The three variants

All three share the dynamics, the reward, the obstacle, and the 200-step limit. They differ only in zone layout and zone semantics.

| Env | Goal | Zones | What a "high" zone does |
| --- | --- | --- | --- |
| `DubinsEnv-v0` | `(2, 2)` | 1 circle, 1 box, no types | n/a, every zone just costs |
| `DubinsRobustEnv-v0` | `(3, -1)` | 2 typed circles, 1 box | **Corrupts the observation** |
| `DubinsEpistemicEnv-v0` | `(3, -1)` | 2 typed circles, 1 box | **Only sets a flag** |

The robust and epistemic envs have the exact same geometry. The contrast between them is the experiment:

- **Robust** injects noise into the returned observation inside a `type: "high"` zone: Gaussian noise at `scale=0.5`, plus a 30% chance of an additional positional bias, then clipped to the observation bounds. Its non-high zone gets milder noise. This is *aleatoric* uncertainty, genuinely present in the data.
- **Epistemic** perturbs nothing at all. Inside a high zone it sets `info['epistemic'] = True` and returns a clean observation. The zones are pure ground-truth labels for where a model *should* report high epistemic uncertainty, which makes them an oracle for scoring an uncertainty estimator rather than a challenge for the controller.

One consequence that is easy to miss: in both typed envs, **cost comes only from the untyped zone**. `cost = 1` is set from `inside_normal`, and the `type: "high"` circles produce noise or a flag but never cost. Safety cost and uncertainty are deliberately decoupled signals, which is what lets MUARL's planner weigh them independently.

**That decoupling holds only if the high zones are disjoint from the cost zones, and until 2026-09-12 they were not.** Both typed envs placed a high circle at `(-2.5,-2.5) r=1.0` nested inside the cost circle at the same centre with `r=1.5`, so `inside_high` implied `cost=1` across 1249 of 25,921 sampled grid states. An agent avoiding that region was unattributable between the uncertainty signal and the cost signal, which is MUARL's A7. The high circle now sits at `(-2.5, 0.5) r=1.0`, clear of every cost zone by 0.5 and identical in both typed envs. `test_zones.py` asserts the property, so a later zone edit cannot quietly reintroduce it.

`info` also carries `distance_sq`, `dist_to_goal`, and `state`, and `reset` returns the full scene description (`uncertainty_zones`, `goal`, `obstacle`) for plotting code to draw against.

## Known defects

These are real and currently present. Fix them if they block you.

The zone geometry moved on 2026-09-12 to fix A7, so **runs already in MUARL's `logs/` are not comparable against anything trained after that**. Treat them as a separate generation rather than as a baseline.

- **Box coordinate order is inconsistent between logic and rendering.** `step` and `_is_in_uncertainty_zone` read `zone['box']` as `(xmin, ymin, xmax, ymax)`. `render` unpacks it as `xmin, xmax, ymin, ymax`. The drawn rectangle therefore does not match the region actually being penalized, in all three envs. Trust the logic, not the picture.
- **Noise in the robust env is unseeded.** It uses the global `np.random` rather than `self.np_random`, so `reset(seed=...)` does not make a robust-env rollout reproducible. Everything else in the env is properly seeded.
- **`info['state']` aliases live state in the base env.** `DubinsEnv5D` assigns `info["state"] = self.state` without copying, and `step` mutates `self.state` in place, so a stored reference silently changes on the next step. The robust and epistemic envs already use `.copy()`. The base env was not updated.
- **`rgb_array` rendering is broken on current matplotlib.** `render` calls `self.fig.canvas.tostring_rgb()`, which matplotlib removed in 3.10. Human-mode rendering is unaffected.
- `render` also appends to the legend on every call for trajectory plots, so long rollouts accumulate duplicate legend entries.

## The MUARL consumer

`/home/alikolling/Development/MUARL` is the only consumer. `muarl/envs/dubins_car_env.py` selects the variant **by substring of the task name**, checking `robust` first, then `epistemic`, then falling through to the base env, and wraps the result in `ActionScaleWrapper`, `ActionRepeat(times=5)`, and `DefaultDictWrapper`. Because of the action repeat, one MUARL step is five env steps, so the effective episode is 40 agent decisions rather than 200.

Two notes on the seam. MUARL's `train.py` picks its trainer by the same substring scheme but against a different list, and its `epistemic` branch imports a module path that does not exist on disk, so that branch is currently broken on the MUARL side rather than here. And `info['epistemic']`, the ground-truth label this package exists to provide, is not currently read anywhere in MUARL. Its `evaluate_unc_methods.py` builds AUROC labels by sampling in-zone and out-of-zone states itself rather than consuming the flag.
