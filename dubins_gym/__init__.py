from gymnasium.envs.registration import register

register(
    id="dubins_gym/DubinsEnv-v0",
    entry_point="dubins_gym.envs:DubinsEnv5D",
    max_episode_steps=200,
)

register(
    id="dubins_gym/DubinsRobustEnv-v0",
    entry_point="dubins_gym.envs:DubinsRobustEnv5D",
    max_episode_steps=200,
)

register(
    id="dubins_gym/DubinsEpistemicEnv-v0",
    entry_point="dubins_gym.envs:DubinsEpistemicEnv5D",
    max_episode_steps=200,
)

# The epistemic env with a deterministic 0.5 m/s sideways drift inside the high zones, so
# withholding them hides dynamics the model has not seen (MUARL PLAN.md, T1).
register(
    id="dubins_gym/DubinsEpistemicDriftEnv-v0",
    entry_point="dubins_gym.envs:DubinsEpistemicEnv5D",
    max_episode_steps=200,
    kwargs={"zone_drift": 0.5},
)
# The same zones, each hiding an obstacle the drift pulls the car toward (MUARL PLAN.md, P5).
register(
    id="dubins_gym/DubinsEpistemicHazardEnv-v0",
    entry_point="dubins_gym.envs:DubinsEpistemicEnv5D",
    max_episode_steps=200,
    kwargs={"zone_drift": 0.5, "layout": "hazard"},
)

# A data frontier: everything north of y = 1.5 is one high zone with a northward drift and three
# hidden obstacles, so withheld data lies on one side only (MUARL PLAN.md, P7).
register(
    id="dubins_gym/DubinsFrontierEnv-v0",
    entry_point="dubins_gym.envs:DubinsEpistemicEnv5D",
    max_episode_steps=200,
    kwargs={"zone_drift": 0.5, "layout": "frontier"},
)

# The frontier with hidden cost zones, a goal inside the strip, and zones of process noise kept
# in the data (MUARL PLAN.md, P8). `noise` is the push std per env step; the ids name it per
# five-step decision, MUARL's action repeat.
register(
    id="dubins_gym/DubinsFrontierExploreEnv-v0",
    entry_point="dubins_gym.envs:DubinsFrontierExploreEnv",
    max_episode_steps=200,
    kwargs={"noise": 0.15 / 5 ** 0.5},
)
register(
    id="dubins_gym/DubinsFrontierExploreLowNoiseEnv-v0",
    entry_point="dubins_gym.envs:DubinsFrontierExploreEnv",
    max_episode_steps=200,
    kwargs={"noise": 0.05 / 5 ** 0.5},
)

# Terrain sensor: train without ice, evaluate with it (MUARL PLAN.md, T1 terrain).
register(
    id="dubins_gym/DubinsTerrainEnv-v0",
    entry_point="dubins_gym.envs:DubinsTerrainEnv",
    max_episode_steps=200,
    kwargs={"ice": False},
)
register(
    id="dubins_gym/DubinsTerrainIceEnv-v0",
    entry_point="dubins_gym.envs:DubinsTerrainEnv",
    max_episode_steps=200,
    kwargs={"ice": True},
)
