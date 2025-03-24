from gymnasium.envs.registration import register

register(
    id='network_env/NetworkSim-v0',
    entry_point='network_env.envs:NetworkEnv',
    # max_episode_steps=300,
)