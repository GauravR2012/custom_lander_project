from gymnasium.envs.registration import register

register(
    id="CustomLunarLander-v0",
    entry_point="custom_lunar_lander:CustomLunarLander",
)
