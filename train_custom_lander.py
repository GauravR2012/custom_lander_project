import gymnasium as gym
import custom_lunar_lander
import register_env
from stable_baselines3 import PPO

env = gym.make("CustomLunarLander-v0")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    clip_range=0.2,
    tensorboard_log="./ppo_custom_lander_logs/"
)

model.learn(total_timesteps=300_000)
model.save("ppo_custom_lander")

env.close()
print("✅ Training completed and model saved.")
