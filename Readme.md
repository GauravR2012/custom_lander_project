#  Reinforcement Learning for Precision Lunar Landing

**Custom OpenAI Gym Environment | Deep Reinforcement Learning | Neuromatch Global Summer School (Deep RL Pod)**

This repository presents a **customized OpenAI Gym LunarLander environment** developed to study **precision-guided landing**, **trajectory optimization**, and **collision avoidance** using **Deep Reinforcement Learning (DRL)** algorithms.  
The environment was optimized and benchmarked using **DQN**, **A2C**, and **PPO**, achieving up to **31% performance improvement** in guided landing accuracy.

---

## Project Overview

This project was carried out as part of the **Deep Reinforcement Learning Pod** at the **Neuromatch Global Summer School**, where a team of six international students collaborated to explore advanced reinforcement learning algorithms for physics-based control problems.

We enhanced the **LunarLander-v2** environment by introducing:
- Custom **reward shaping** for smooth descent and stable landings.  
- Fine-tuned **action space** for more precise thrust control.  
- Additional **penalty terms** for off-angle landings and high-velocity impacts.  
- New **collision avoidance** and **trajectory optimization** components.

---

## Objectives

1. Develop a **custom LunarLander environment** focusing on control precision.  
2. Compare and evaluate **DQN**, **A2C**, and **PPO** for stability and convergence.  
3. Optimize landing trajectories while minimizing collision probability.  
4. Visualize policy behavior and analyze landing dynamics quantitatively.

---

##  Key Results

| Algorithm | Average Reward | Improvement vs Baseline | Stability |
|------------|----------------|--------------------------|------------|
| DQN        | 216            | +18.5%                   | Moderate   |
| A2C        | 237            | +23.8%                   | High       |
| **PPO**    | **264**        | **+31.0%**               | **Very High** |

The **PPO agent** consistently achieved smoother and safer landings with optimal fuel efficiency and minimal oscillation in trajectory.

---

## Repository Structure


📦 lunar-lander-rl
│
├── lunarlander_custom/
│ ├── init.py
│ └── lunarlander_custom_env.py # Customized Gym environment
│
├── train_dqn.py # DQN training script
├── train_a2c.py # A2C training script
├── train_ppo.py # PPO training script
│
├── requirements.txt
├── setup.py # Registers custom Gym environment
├── README.md
└── results/
├── logs/
├── videos/
└── plots/

