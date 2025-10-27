from stable_baselines3 import PPO
import gymnasium as gym
import custom_lunar_lander
import register_env

env = gym.make("CustomLunarLander-v0", render_mode="human")
model = PPO.load("ppo_custom_lander", env=env)

obs, _ = env.reset()
for _ in range(1500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
