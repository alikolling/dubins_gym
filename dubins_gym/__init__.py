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
