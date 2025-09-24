import copy

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict


def _convert_space(space):
    if isinstance(space, Box):
        space = Box(space.low, space.high, space.shape)
    elif isinstance(space, Dict):
        for k, v in space.spaces.items():
            space.spaces[k] = _convert_space(v)
        space = Dict(space.spaces)
    else:
        raise NotImplementedError
    return space


def _convert(x):
    if isinstance(x, np.ndarray):
        if x.dtype == np.float64:
            return x.astype(np.float32)
        else:
            return x
    elif isinstance(x, dict):
        x = copy.copy(x)
        for k, v in x.items():
            x[k] = _convert(v)
        return x


class SinglePrecisionObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_space = copy.deepcopy(self.env.observation_space)
        self.observation_space = _convert_space(obs_space)

    def observation(self, observation):
        return _convert(observation)


class SinglePrecisionAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        action_space = copy.deepcopy(self.env.action_space)
        self.action_space = _convert_space(action_space)

    def action(self, action):
        return _convert(action)
