import numpy as np
from typing import List, Optional

import gymnasium as gym
from gymnasium import spaces
import pygame


class DIEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        mass=1.,
        friction=0.,
        action_factor: float = 1.,
        action_factor_0: float = 1.,
        action_factor_1: float = 1.,
        action_disturbance: float = 0.,
        reward_weights: List[float] = [1.0, 0.1],
        x_goal: float = 0.,
        y_goal: float = 0.,
        reward_type: str = "dense",
        start_position_is_center: bool = False,
    ):
        self.mass = mass
        self.friction = friction
        action_factor_i = np.array([action_factor_0, action_factor_1], dtype=np.float32)
        self.action_factor = action_factor * action_factor_i
        self.action_disturbance = action_disturbance
        self.reward_weights = np.array(reward_weights, dtype=np.float32)

        self.dt = 0.1

        self.goal_position = np.array([x_goal, y_goal], dtype=np.float32)
        self.goal_threshold = 0.1
        self.reward_type = reward_type

        self.start_position_is_center = start_position_is_center
        if start_position_is_center:
            assert x_goal != 0.0 or y_goal != 0.0

        self.min_position = np.array([-1, -1], dtype=np.float32)
        self.max_position = np.array([1, 1], dtype=np.float32)
        self.min_velocity = np.array([-1, -1], dtype=np.float32)
        self.max_velocity = np.array([1, 1], dtype=np.float32)

        self.min_action = np.array([-1, -1], dtype=np.float32)  # Force in x and y direction
        self.max_action = np.array([1, 1], dtype=np.float32)

        self.low_state = np.concatenate([self.min_position, self.min_velocity], dtype=np.float32)
        self.high_state = np.concatenate([self.max_position, self.max_velocity], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float32
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window_size = 512
        self.window = None
        self.clock = None

        self._context = {
            "mass": mass,
            "friction": friction,
            "action_factor": action_factor,
            "action_factor_0": action_factor_0,
            "action_factor_1": action_factor_1,
            "action_disturbance": action_disturbance,
            "reward_weights": reward_weights,
            "x_goal": x_goal,
            "y_goal": y_goal,
        }
        self._context_low = {
            "mass": 0.0,
            "friction": 0.0,
            "action_factor": -np.inf,
            "action_factor_0": -np.inf,
            "action_factor_1": -np.inf,
            "action_disturbance": -np.inf,
            "reward_weights": [0.0, 0.0],
            "x_goal": -1.0,
            "y_goal": -1.0,
        }
        self._context_high = {
            "mass": np.inf,
            "friction": np.inf,
            "action_factor": np.inf,
            "action_factor_0": np.inf,
            "action_factor_1": np.inf,
            "action_disturbance": np.inf,
            "reward_weights": [np.inf, np.inf],
            "x_goal": 1.0,
            "y_goal": 1.0,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.velocity = np.array([0, 0], dtype=np.float32)

        if self.start_position_is_center:
            self.position = np.array([0, 0], dtype=np.float32)
        else:
            # Sample agent's position far away from the center corresponding to both x and y component.
            while True:
                self.position = self.np_random.uniform(low=-1, high=1, size=2).astype(np.float32)
                if not np.logical_and(self.position >= -0.8, self.position <= 0.8).any():
                    break

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        force = np.clip(
            np.array(action, dtype=np.float32) * self.action_factor + self.action_disturbance,
            self.min_action,
            self.max_action
        )

        # Equation of motion, Euler discretization
        self.position += self.velocity * self.dt
        self.velocity = self.velocity * (1 - self.dt * self.friction / self.mass) + force / self.mass * self.dt

        # Comply with bounds
        self.position = np.clip(self.position, self.min_position, self.max_position)
        self.velocity = np.clip(self.velocity, self.min_velocity, self.max_velocity)

        distance, distance_x, distance_y = self._get_distance()
        if self.reward_type == "sparse":
            scalar_reward = float(distance <= self.goal_threshold)
        elif self.reward_type == "dense":
            reward = np.array([-distance ** 2, -np.linalg.norm(force, ord=2) ** 2], dtype=np.float32)
            scalar_reward = np.dot(reward, self.reward_weights)
        else:
            raise NotImplementedError()

        observation = self._get_obs()
        info = {"distance": distance, "distance_x": distance_x, "distance_y": distance_y}

        if self.render_mode == "human":
            self._render_frame()

        return observation, scalar_reward, False, False, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                'e.g. gym.make(render_mode="rgb_array")'
            )
            return

        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        return np.concatenate([self.position, self.velocity], dtype=np.float32)

    def _get_distance(self):
        distance = np.linalg.norm(self.position - self.goal_position, ord=2)
        distance_x = abs(self.position[0] - self.goal_position[0])
        distance_y = abs(self.position[1] - self.goal_position[1])
        return distance, distance_x, distance_y

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Grid
        pygame.draw.line(
            canvas,
            0,
            self._coords2pygame((0, self.window_size / 2)),
            self._coords2pygame((self.window_size, self.window_size / 2)),
            width=3,
        )
        pygame.draw.line(
            canvas,
            0,
            self._coords2pygame((self.window_size / 2, 0)),
            self._coords2pygame((self.window_size / 2, self.window_size)),
            width=3,
        )

        # Draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            self._coords2pygame(
                tuple(((self.goal_position + 1) * (self.window_size / 2)).tolist())
            ),
            self.window_size * 0.05,
        )

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._coords2pygame(
                tuple(((self.position + 1) * (self.window_size / 2)).tolist())
            ),
            self.window_size * 0.01,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _coords2pygame(self, coords, height=None):
        # Convert coordinates into pygame coordinates (lower-left -> top left).
        if height is None:
            height = self.window_size
        return (coords[0], height - coords[1])


if __name__ == "__main__":
    env = DIEnv(render_mode="human")
    _, _ = env.reset(seed=0)
    env.step([1, 1])
