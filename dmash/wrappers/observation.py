import copy
from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.core import ActType, ObsType, WrapperObsType


class MaskObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Multiply the mask array to the observation.

    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        mask: np.ndarray = None,
    ):
        """A wrapper for adding information on the active action wrapper to the observation.

        Args:
            env: The environment to wrap
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        if mask is None:
            mask = np.ones(self.observation_space.shape)
        assert mask.shape == self.observation_space.shape, "Provide a valid mask for the observation."
        self.mask = mask

        assert isinstance(env.observation_space, Box), "Only Box obs_space supported."

        if isinstance(env.observation_space, Dict):
            new_observation_space = copy.deepcopy(env.observation_space)
            new_observation_space["context"] = Box(
                low=0,
                high=1,
                shape=self.mask.shape,
                dtype=np.int8
            )
        elif isinstance(env.observation_space, Box):
            new_observation_space = Box(
                low=np.concatenate([env.observation_space.low, np.ones_like(self.mask) * min(self.mask)]),
                high=np.concatenate([env.observation_space.high, np.ones_like(self.mask) * min(self.mask)]),
                shape=(env.observation_space.shape[0] + self.mask.shape[0],),
                dtype=env.observation_space.dtype,
            )
        else:
            raise NotImplementedError()
        self.observation_space = new_observation_space
        self._env = env

    def observation(self, observation: ObsType) -> Any:
        """Apply function to the observation."""
        return np.concatenate([observation * self.mask, self.mask])

    def set_mask(self, mask: np.ndarray):
        assert mask.shape[0] == self.observation_space.shape[0] / 2, "Provide a valid mask for the observation."
        self.mask = mask


class ContextAwareObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Adds a context array to the observation.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        noisy_context_lim: float = 0.0,
        context_aware_varied: bool = False,
        contexts: dict = {},
    ):
        """A wrapper for adding context information to the observation.

        Args:
            env: The environment to wrap
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        assert hasattr(env.unwrapped, "_context"), "Env does not contain context."
        assert hasattr(env.unwrapped, "_context_low"), "Env does not contain context."
        assert hasattr(env.unwrapped, "_context_high"), "Env does not contain context."

        _context = env.unwrapped._context
        _context_low = env.unwrapped._context_low
        _context_high = env.unwrapped._context_high
        if context_aware_varied:
            _context = contexts
            _context_low = {k: _context_low[k] for k in _context.keys()}
            _context_high = {k: _context_high[k] for k in _context.keys()}
        self.context = self._context2array(_context)
        context_low = self._context2array(_context_low)
        context_high = self._context2array(_context_high)

        self.noisy_context_lim = noisy_context_lim

        if isinstance(env.observation_space, Dict):
            new_observation_space = copy.deepcopy(env.observation_space)
            new_observation_space["context"] = Box(
                low=context_low,
                high=context_high,
                shape=self.context.shape,
                dtype=np.float64
            )
        elif isinstance(env.observation_space, Box):
            new_observation_space = Box(
                low=np.concatenate(
                    [env.observation_space.low, context_low]
                ),
                high=np.concatenate(
                    [env.observation_space.high, context_high]
                ),
                shape=(env.observation_space.shape[0] + self.context.shape[0],),
                dtype=env.observation_space.dtype,
            )
        else:
            raise NotImplementedError()
        self.observation_space = new_observation_space

    def _context2array(self, context: Dict) -> np.ndarray:
        return np.array(self._flatten_list(list(context.values())))

    def _flatten_list(self, x):
        if not isinstance(x, list):
            return [x]
        return [xss for xs in x for xss in self._flatten_list(xs)]

    def observation(self, observation: ObsType) -> Any:
        if self.noisy_context_lim > 0.0:
            noise = 1.0 + (self.np_random.random(self.context.size) * 2 - 1) * self.noisy_context_lim
        else:
            noise = 1.0
        if isinstance(self.env.observation_space, Dict):
            observation["context"] = self.context * noise
            return observation
        elif isinstance(self.env.observation_space, Box):
            return np.concatenate([observation, self.context * noise])
        else:
            raise NotImplementedError()


class OneHotAwareObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Adds a context array to the observation.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        task_idx: int,
        num_tasks: int
    ):
        """A wrapper for adding context information to the observation.

        Args:
            env: The environment to wrap
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        one_hot_high = np.ones(num_tasks)
        one_hot_low = np.zeros(num_tasks)

        self.one_hot = np.zeros(num_tasks)
        self.one_hot[task_idx] = 1.0

        if isinstance(env.observation_space, Dict):
            new_observation_space = copy.deepcopy(env.observation_space)
            new_observation_space["context"] = Box(
                low=one_hot_low,
                high=one_hot_high,
                shape=self.one_hot.shape,
                dtype=env.observation_space.dtype
            )
        elif isinstance(env.observation_space, Box):
            new_observation_space = Box(
                low=np.concatenate(
                    [env.observation_space.low, one_hot_low]
                ),
                high=np.concatenate(
                    [env.observation_space.high, one_hot_high]
                ),
                shape=(env.observation_space.shape[0] + self.one_hot.shape[0],),
                dtype=env.observation_space.dtype,
            )
        else:
            raise NotImplementedError()
        self.observation_space = new_observation_space

    def observation(self, observation: ObsType) -> Any:
        if isinstance(self.env.observation_space, Dict):
            observation["context"] = self.one_hot
            return observation
        elif isinstance(self.env.observation_space, Box):
            return np.concatenate([observation, self.one_hot])
        else:
            raise NotImplementedError()


class NoisyObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Adds multiplicative noise to the observation.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        noisy_lim: float = 0.0,
    ):
        """A wrapper for adding noise to the observation.

        Args:
            env: The environment to wrap
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        assert isinstance(self.env.observation_space, Box)
        self.noisy_lim = noisy_lim

    def observation(self, observation: ObsType) -> Any:
        if self.noisy_lim > 0.0:
            noise = 1.0 + (np.random.rand(self.observation_space.shape[0]) * 2 - 1) * self.noisy_lim
        else:
            noise = 1.0
        return observation * noise


class DistractedObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Concatenates distractor variables to the observation.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        num_distractors: int = 0,
        fixed_distractors: bool = False,
    ):
        """A wrapper for concatenating distractors to the observation.

        Args:
            env: The environment to wrap
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        assert isinstance(self.env.observation_space, Box)
        self.num_distractors = num_distractors

        if isinstance(env.observation_space, Dict):
            new_observation_space = copy.deepcopy(env.observation_space)
            new_observation_space["distractors"] = Box(
                low=np.zeros(self.num_distractors),
                high=np.ones(self.num_distractors),
                shape=(self.num_distractors,),
                dtype=env.observation_space.dtype
            )
        elif isinstance(env.observation_space, Box):
            new_observation_space = Box(
                low=np.concatenate(
                    [env.observation_space.low, np.zeros(self.num_distractors)]
                ),
                high=np.concatenate(
                    [env.observation_space.high, np.ones(self.num_distractors)]
                ),
                shape=(env.observation_space.shape[0] + self.num_distractors,),
                dtype=env.observation_space.dtype,
            )
        else:
            raise NotImplementedError()
        self.observation_space = new_observation_space

        self.fixed_distractors = fixed_distractors
        self.distractors = np.random.rand(self.num_distractors) if fixed_distractors else None

    def observation(self, observation: ObsType) -> Any:
        if not self.fixed_distractors:
            self.distractors = np.random.rand(self.num_distractors)
        if isinstance(self.env.observation_space, Dict):
            observation["distractors"] = self.distractors
            return observation
        elif isinstance(self.env.observation_space, Box):
            return np.concatenate([observation, self.distractors])
        else:
            raise NotImplementedError()
