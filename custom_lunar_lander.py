import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np
import Box2D

class CustomLunarLander(LunarLander):
    """
    Custom Gym Lunar Lander for trajectory optimization and collision avoidance.
    """

    def __init__(self, continuous=True, add_obstacles=True):
        super().__init__(continuous=continuous)
        self.add_obstacles = add_obstacles
        self.obstacles = []

        # Physics tweaks for realism
        self.world.gravity = (0, -11)
        self.main_engine_power = 14.0
        self.side_engine_power = 0.8

        # Continuous control: [side_thrust, main_thrust]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Add obstacles to the terrain
        if self.add_obstacles:
            self._create_obstacles()

    def _create_obstacles(self):
        """Add static obstacles to simulate collision avoidance."""
        ground_y = self.helipad_y
        obstacle_positions = [(-0.8, ground_y + 0.15), (0.8, ground_y + 0.25)]
        for (x, y) in obstacle_positions:
            obstacle = self.world.CreateStaticBody(
                position=(x, y),
                shapes=Box2D.b2PolygonShape(box=(0.15, 0.15))
            )
            self.obstacles.append(obstacle)

    def step(self, action):
        # Random wind disturbance for realism
        wind = self.np_random.uniform(-1, 1) * 0.1
        self.lander.ApplyForceToCenter((wind, 0), True)

        state, reward, terminated, truncated, info = super().step(action)

        x, y, vx, vy, angle, angular_vel, leg1, leg2 = state

        # --- Reward shaping for trajectory optimization ---
        smoothness_penalty = 0.4 * (abs(vx) + abs(vy))
        angle_penalty = 0.2 * abs(angle)
        fuel_penalty = 0.05 * np.sum(np.abs(action))

        # --- Collision avoidance penalty ---
        safe_zone = abs(x) < 0.3 and abs(angle) < 0.2 and y < 0.2
        if not safe_zone and y < 0.1:
            reward -= 20.0

        for obs in self.obstacles:
            for contact in self.lander.contacts:
                if contact.contact.shape in obs.fixtures:
                    reward -= 30.0
                    terminated = True

        reward -= (smoothness_penalty + angle_penalty + fuel_penalty)

        return np.array(state, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        """Render obstacles along with the lander."""
        result = super().render()
        if self.add_obstacles and hasattr(self, 'viewer'):
            for obs in self.obstacles:
                self.viewer.draw_polygon(
                    [(obs.position[0] - 0.15, obs.position[1] - 0.15),
                     (obs.position[0] - 0.15, obs.position[1] + 0.15),
                     (obs.position[0] + 0.15, obs.position[1] + 0.15),
                     (obs.position[0] + 0.15, obs.position[1] - 0.15)],
                    color=(0.8, 0.2, 0.2)
                )
        return result
