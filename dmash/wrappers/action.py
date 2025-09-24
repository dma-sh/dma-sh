import collections
import copy
from typing import Any, Callable, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Space
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType


class SleepingTransformAction(
    gym.Wrapper[ObsType, WrapperActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Applies an inactive function to the ``action`` before passing the modified value to the environment ``step`` function.

    Adapted from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/transform_action.py
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[WrapperActType], ActType],
        action_space: Space[WrapperActType] | None,
        toggle_at_step: int | list[int] | float | None = None,
        toggle_at_episode: int | list[int] | float | None = None
    ):
        """Initialize SleepingTransformAction.

        Args:
            env: The environment to wrap.
            func: Function to apply to the :meth:`step`'s ``action``
            action_space: The updated action space of the wrapper given the function.
            toggle_at_step: Decides whether to apply transformation. If type is int, it's considered
            True for all steps ongoing. If type is list[int], that represents a list of steps at
            which it's alternately toggled True/False. If type is float, it's considered True for
            that frequency of steps.
            toggle_at_episode: Decides whether to apply transformation. If type is int, it's
            considered True for all episodes ongoing. If type is list[int], that represents a list
            of episodes at which it's alternately toggled True/False. If type is float, it's
            considered True for that frequency of episodes.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            func=func,
            action_space=action_space,
            toggle_at_step=toggle_at_step,
            toggle_at_episode=toggle_at_episode
        )
        gym.Wrapper.__init__(self, env)

        if action_space is not None:
            self.action_space = action_space

        self.step_count = 0
        self.episode_count = 0

        self.func = func

        assert (toggle_at_step is None) != (toggle_at_episode is None), (
            "Make sure either toggle_at_step or toggle_at_episode is defined, not both."
        )
        if isinstance(toggle_at_step, float):
            assert 0.0 <= toggle_at_step <= 1.0, "Toggle frequency has to be between 0.0 and 1.0."
        elif isinstance(toggle_at_step, int):
            toggle_at_step = [toggle_at_step]
        elif isinstance(toggle_at_step, list):
            toggle_at_step = sorted(toggle_at_step)
        if isinstance(toggle_at_episode, float):
            assert 0.0 <= toggle_at_episode <= 1.0, "Toggle frequency has to be between 0.0 and 1.0."
        elif isinstance(toggle_at_episode, int):
            toggle_at_episode = [toggle_at_episode]
        elif isinstance(toggle_at_episode, list):
            toggle_at_episode = sorted(toggle_at_episode)

        self._toggle_at_step = toggle_at_step
        self._toggle_at_episode = toggle_at_episode

    def step(
        self, action: WrapperActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runsthe :attr:`env` :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        obs, reward, terminated, truncated, info = self.env.step(self.action(action))
        self.step_count += 1
        if truncated or terminated:
            self.episode_count += 1
        return obs, reward, terminated, truncated, info

    def action(self, action: WrapperActType) -> ActType:
        """Apply function to action."""
        return self.func(action) if self.is_active() else action

    def is_active(self):
        if self._toggle_at_step is not None:
            if isinstance(self._toggle_at_step, list):
                return sum([self.step_count >= t for t in self._toggle_at_step]) % 2 == 1
            elif isinstance(self._toggle_at_step, float):
                return (self.step_count * self._toggle_at_step) % 1 == 0
            else:
                raise NotImplementedError()
        elif self._toggle_at_episode is not None:
            if isinstance(self._toggle_at_episode, list):
                return sum([self.episode_count >= t for t in self._toggle_at_episode]) % 2 == 1
            elif isinstance(self._toggle_at_episode, float):
                return (self.episode_count * self._toggle_at_episode) % 1 == 0
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()


class InvertAction(
    SleepingTransformAction[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Inverts one or all dimensions of ``action`` which is passed to ``step``.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from ubada import InvertAction
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> env = InvertAction(env, dim=0, toggle_at_step=0)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([0.5, 0.5, 0.5], dtype=np.float32))
        ... # Executes the action np.array([-0.5, 0.5, 0.5]) in the base environment
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        dim: int | None = None,
        toggle_at_step: int | list[int] | float | None = None,
        toggle_at_episode: int | list[int] | float | None = None
    ):
        """A wrapper for inverting one or all dimensions of the continuous action.

        Args:
            env: The environment to wrap.
            dim: Action dimension that is transformed. None corresponds to all dimensions.
            toggle_at_step: Decides whether to apply transformation. If type is int, it's considered
            True for all steps ongoing. If type is list[int], that represents a list of steps at
            which it's alternately toggled True/False. If type is float, it's considered True for
            that frequency of steps.
            toggle_at_episode: Decides whether to apply transformation. If type is int, it's
            considered True for all episodes ongoing. If type is list[int], that represents a list
            of episodes at which it's alternately toggled True/False. If type is float, it's
            considered True for that frequency of episodes.
        """
        assert isinstance(env.action_space, Box)

        def _action_func(action):
            _action = action.copy()
            if dim is not None:
                _action[dim] *= -1
            else:
                _action *= -1
            return _action

        gym.utils.RecordConstructorArgs.__init__(self)
        SleepingTransformAction.__init__(
            self,
            env=env,
            func=_action_func,
            action_space=None,
            toggle_at_step=toggle_at_step,
            toggle_at_episode=toggle_at_episode
        )


class NoiseAction(
    SleepingTransformAction[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Adds gaussian noise to one or all dimensions of ``action`` which is passed to ``step``.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from ubada import NoiseAction
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> env = NoiseAction(env, dim=0, toggle_at_step=0, value=0.3)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([0.5, 0.5, 0.5], dtype=np.float32))
        ... # Executes the action np.array([0.33258423, 0.5, 0.5]) in the base environment
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        dim: int | None = None,
        value: float = 0.3,
        toggle_at_step: int | list[int] | float | None = None,
        toggle_at_episode: int | list[int] | float | None = None
    ):
        """A wrapper for adding gaussian noise to one or all dimensions of the continuous action.

        Args:
            env: The environment to wrap.
            dim: Action dimension that is transformed. None corresponds to all dimensions.
            value: Standard deviation of the gaussian distribution.
            toggle_at_step: Decides whether to apply transformation. If type is int, it's considered
            True for all steps ongoing. If type is list[int], that represents a list of steps at
            which it's alternately toggled True/False. If type is float, it's considered True for
            that frequency of steps.
            toggle_at_episode: Decides whether to apply transformation. If type is int, it's
            considered True for all episodes ongoing. If type is list[int], that represents a list
            of episodes at which it's alternately toggled True/False. If type is float, it's
            considered True for that frequency of episodes.
        """
        assert isinstance(env.action_space, Box)

        def _action_func(action):
            _action = action.copy()
            if dim is not None:
                _action[dim] += np.random.normal(loc=0.0, scale=value)
            else:
                _action += np.random.normal(loc=0.0, scale=value, size=action.shape)
            return _action

        gym.utils.RecordConstructorArgs.__init__(self)
        SleepingTransformAction.__init__(
            self,
            env=env,
            func=_action_func,
            action_space=None,
            toggle_at_step=toggle_at_step,
            toggle_at_episode=toggle_at_episode
        )


class SineNoiseAction(
    SleepingTransformAction[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Adds gaussian noise overlaid with a sine to one or all dimensions of ``action`` which is
    passed to ``step``.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from ubada import SineNoiseAction
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> env = SineNoiseAction(env, dim=0, toggle_at_step=0, value=0.3)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([0.5, 0.5, 0.5], dtype=np.float32))
        ... # Executes the action np.array([0.24253476, 0.5, 0.5]) in the base environment
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        dim: int | None = None,
        value: float = 0.3,
        toggle_at_step: int | list[int] | float | None = None,
        toggle_at_episode: int | list[int] | float | None = None
    ):
        """A wrapper for adding gaussian noise overlaid with a sine to one or all dimensions of the
        continuous action.

        Args:
            env: The environment to wrap.
            dim: Action dimension that is transformed. None corresponds to all dimensions.
            value: Standard deviation of the gaussian distribution and the impact of the sine curve.
            toggle_at_step: Decides whether to apply transformation. If type is int, it's considered
            True for all steps ongoing. If type is list[int], that represents a list of steps at
            which it's alternately toggled True/False. If type is float, it's considered True for
            that frequency of steps.
            toggle_at_episode: Decides whether to apply transformation. If type is int, it's
            considered True for all episodes ongoing. If type is list[int], that represents a list
            of episodes at which it's alternately toggled True/False. If type is float, it's
            considered True for that frequency of episodes.
        """
        assert isinstance(env.action_space, Box)

        def _action_func(action):
            _action = action.copy()
            if dim is not None:
                _action[dim] += (
                    np.random.normal(loc=0.0, scale=value)
                    + np.sin(self.step_count) * value
                )
            else:
                _action += (
                    np.random.normal(loc=0.0, scale=value, size=action.shape)
                    + np.sin(self.step_count) * value
                )
            return _action

        gym.utils.RecordConstructorArgs.__init__(self)
        SleepingTransformAction.__init__(
            self,
            env=env,
            func=_action_func,
            action_space=None,
            toggle_at_step=toggle_at_step,
            toggle_at_episode=toggle_at_episode
        )


class OffsetAction(
    SleepingTransformAction[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Adds offset to one or all dimensions of ``action`` which is passed to ``step``.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from ubada import OffsetAction
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> env = OffsetAction(env, dim=0, toggle_at_step=0, value=0.3)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([0.5, 0.5, 0.5], dtype=np.float32))
        ... # Executes the action np.array([0.8, 0.5, 0.5]) in the base environment
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        dim: int | None = None,
        value: float = 0.3,
        toggle_at_step: int | list[int] | float | None = None,
        toggle_at_episode: int | list[int] | float | None = None
    ):
        """A wrapper for adding an offset to one or all dimensions of the continuous action.

        Args:
            env: The environment to wrap.
            dim: Action dimension that is transformed. None corresponds to all dimensions.
            value: Offset which is added to the action (dimension).
            toggle_at_step: Decides whether to apply transformation. If type is int, it's considered
            True for all steps ongoing. If type is list[int], that represents a list of steps at
            which it's alternately toggled True/False. If type is float, it's considered True for
            that frequency of steps.
            toggle_at_episode: Decides whether to apply transformation. If type is int, it's
            considered True for all episodes ongoing. If type is list[int], that represents a list
            of episodes at which it's alternately toggled True/False. If type is float, it's
            considered True for that frequency of episodes.
        """
        assert isinstance(env.action_space, Box)

        def _action_func(action):
            _action = action.copy()
            if dim is not None:
                _action[dim] += value
            else:
                _action += value
            return _action

        gym.utils.RecordConstructorArgs.__init__(self)
        SleepingTransformAction.__init__(
            self,
            env=env,
            func=_action_func,
            action_space=None,
            toggle_at_step=toggle_at_step,
            toggle_at_episode=toggle_at_episode
        )


class ScaleAction(
    SleepingTransformAction[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Scales one or all dimensions of ``action`` which is passed to ``step``.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from ubada import ScaleAction
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> env = ScaleAction(env, dim=0, toggle_at_step=0, value=0.5)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([0.5, 0.5, 0.5], dtype=np.float32))
        ... # Executes the action np.array([0.25, 0.5, 0.5]) in the base environment
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        dim: int | None = None,
        value: float = 0.3,
        toggle_at_step: int | list[int] | float | None = None,
        toggle_at_episode: int | list[int] | float | None = None
    ):
        """A wrapper for scaling one or all dimensions of the continuous action.

        Args:
            env: The environment to wrap.
            dim: Action dimension that is transformed. None corresponds to all dimensions.
            value: Scaling factor which is multiplied with the action (dimension).
            toggle_at_step: Decides whether to apply transformation. If type is int, it's considered
            True for all steps ongoing. If type is list[int], that represents a list of steps at
            which it's alternately toggled True/False. If type is float, it's considered True for
            that frequency of steps.
            toggle_at_episode: Decides whether to apply transformation. If type is int, it's
            considered True for all episodes ongoing. If type is list[int], that represents a list
            of episodes at which it's alternately toggled True/False. If type is float, it's
            considered True for that frequency of episodes.
        """
        assert isinstance(env.action_space, Box)

        def _action_func(action):
            _action = action.copy()
            if dim is not None:
                _action[dim] *= value
            else:
                _action *= value
            return _action

        gym.utils.RecordConstructorArgs.__init__(self)
        SleepingTransformAction.__init__(
            self,
            env=env,
            func=_action_func,
            action_space=None,
            toggle_at_step=toggle_at_step,
            toggle_at_episode=toggle_at_episode
        )


class RepeatAction(
    SleepingTransformAction[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Repeats one or all dimensions of ``action`` which is passed to ``step``.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from ubada import RepeatAction
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> env = RepeatAction(env, dim=0, toggle_at_step=0, value=0.5)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([0.5, 0.5, 0.5], dtype=np.float32))
        ... # Executes the action np.array([0.5, 0.5, 0.5]) in the base environment
        >>> _ = env.step(np.array([0.2, 0.2, 0.2], dtype=np.float32))
        ... # Executes the action np.array([0.5, 0.2, 0.2]) in the base environment
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        dim: int | None = None,
        value: float = 0.3,
        num_repeat: int = 4,
        toggle_at_step: int | list[int] | float | None = None,
        toggle_at_episode: int | list[int] | float | None = None
    ):
        """A wrapper for randomly repeating one or all dimensions of the action for a number of steps.

        Args:
            env: The environment to wrap.
            dim: Action dimension that is transformed. None corresponds to all dimensions.
            value: Percentage of actions (dimension) are repeated.
            num_repeat: Number of steps for which the action (dimension) is repeated.
            toggle_at_step: Decides whether to apply transformation. If type is int, it's considered
            True for all steps ongoing. If type is list[int], that represents a list of steps at
            which it's alternately toggled True/False. If type is float, it's considered True for
            that frequency of steps.
            toggle_at_episode: Decides whether to apply transformation. If type is int, it's
            considered True for all episodes ongoing. If type is list[int], that represents a list
            of episodes at which it's alternately toggled True/False. If type is float, it's
            considered True for that frequency of episodes.
        """

        assert 0.0 <= value <= 1.0, "Value for RepeatAction wrapper should be between 0.0 and 1.0."
        self._num_repeat = num_repeat
        self._repeat_stack = collections.deque(maxlen=self._num_repeat)

        def _action_func(action):
            _action = action.copy()
            if len(self._repeat_stack) == 0:
                if value > np.random.random():
                    for _ in range(self._num_repeat):
                        self._repeat_stack.append(_action)
                else:
                    return _action
            if dim is not None:
                _action[dim] = self._repeat_stack.pop()[dim]
            else:
                _action = self._repeat_stack.pop()
            return _action

        gym.utils.RecordConstructorArgs.__init__(self)
        SleepingTransformAction.__init__(
            self,
            env=env,
            func=_action_func,
            action_space=None,
            toggle_at_step=toggle_at_step,
            toggle_at_episode=toggle_at_episode
        )


class ZeroAction(
    SleepingTransformAction[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Zeros one or all dimensions of ``action`` which is passed to ``step``.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from ubada import ZeroAction
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> env = ZeroAction(env, dim=0, toggle_at_step=0, value=0.5)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([0.5, 0.5, 0.5], dtype=np.float32))
        ... # Executes the action np.array([0.5, 0.5, 0.5]) in the base environment
        >>> _ = env.step(np.array([0.2, 0.2, 0.2], dtype=np.float32))
        ... # Executes the action np.array([0.0, 0.2, 0.2]) in the base environment
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        dim: int | None = None,
        value: float = 0.3,
        num_repeat: int = 4,
        toggle_at_step: int | list[int] | float | None = None,
        toggle_at_episode: int | list[int] | float | None = None
    ):
        """A wrapper for randomly zeroing one or all dimensions of the continous action for a
        number of steps.

        Args:
            env: The environment to wrap.
            dim: Action dimension that is transformed. None corresponds to all dimensions.
            value: Percentage of actions (dimension) are zeroed .
            num_repeat: Number of steps for which the action (dimension) is zeroed.
            toggle_at_step: Decides whether to apply transformation. If type is int, it's considered
            True for all steps ongoing. If type is list[int], that represents a list of steps at
            which it's alternately toggled True/False. If type is float, it's considered True for
            that frequency of steps.
            toggle_at_episode: Decides whether to apply transformation. If type is int, it's
            considered True for all episodes ongoing. If type is list[int], that represents a list
            of episodes at which it's alternately toggled True/False. If type is float, it's
            considered True for that frequency of episodes.
        """
        assert isinstance(env.action_space, Box)

        assert 0.0 <= value <= 1.0, "Value for ZeroAction wrapper should be between 0.0 and 1.0."
        self._num_repeat = num_repeat
        self._zero_stack = collections.deque(maxlen=self._num_repeat)

        def _action_func(action):
            _action = action.copy()
            if len(self._zero_stack) == 0:
                if value > np.random.random():
                    for _ in range(self._num_repeat):
                        self._zero_stack.append(0)
                else:
                    return _action
            if dim is not None:
                _action[dim] = self._zero_stack.pop()
            else:
                _action[:] = self._zero_stack.pop()
            return _action

        gym.utils.RecordConstructorArgs.__init__(self)
        SleepingTransformAction.__init__(
            self,
            env=env,
            func=_action_func,
            action_space=None,
            toggle_at_step=toggle_at_step,
            toggle_at_episode=toggle_at_episode
        )


class SwapAction(
    SleepingTransformAction[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Swaps one or shuffles all dimensions of ``action`` which is passed to ``step``.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from ubada import ScaleAction
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> env = ScaleAction(env, dim=0, toggle_at_step=0)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([0.1, 0.3, 0.5], dtype=np.float32))
        ... # Executes the action np.array([0.3, 0.1, 0.5]) in the base environment
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        dim: int | None = None,
        toggle_at_step: int | list[int] | float | None = None,
        toggle_at_episode: int | list[int] | float | None = None
    ):
        """A wrapper for swapping one dimension with another (randomly picked) or shuffling all
        dimensions of the action. Changed order of actions dims is kept.

        Args:
            env: The environment to wrap.
            dim: Action dimension that is swapped. None corresponds to shuffleing all dimensions.
            toggle_at_step: Decides whether to apply transformation. If type is int, it's considered
            True for all steps ongoing. If type is list[int], that represents a list of steps at
            which it's alternately toggled True/False. If type is float, it's considered True for
            that frequency of steps.
            toggle_at_episode: Decides whether to apply transformation. If type is int, it's
            considered True for all episodes ongoing. If type is list[int], that represents a list
            of episodes at which it's alternately toggled True/False. If type is float, it's
            considered True for that frequency of episodes.
        """

        assert env.action_space.shape[0] >= 2, "Swapping only possible for more than one action dim."
        self._shuffle_index = np.arange(env.action_space.shape[0])
        if dim is not None:
            swap_dim_with = np.random.choice(np.delete(self._shuffle_index, dim))
            self._shuffle_index[swap_dim_with], self._shuffle_index[dim] = dim, swap_dim_with
        else:
            np.random.shuffle(self._shuffle_index)

        def _action_func(action):
            _action = action.copy()
            _action = _action[self._shuffle_index]
            return _action

        gym.utils.RecordConstructorArgs.__init__(self)
        SleepingTransformAction.__init__(
            self,
            env=env,
            func=_action_func,
            action_space=None,
            toggle_at_step=toggle_at_step,
            toggle_at_episode=toggle_at_episode
        )


class DynamicsHintObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Adds a one-hot identifier corresponding to the active action wrapper the the observation.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from ubada import InvertAction, DynamicsHintObservation
        >>> env = gym.make("Pendulum-v1", disable_env_checker=True)
        >>> env.observation_space.shape
        (3,)
        >>> env = InvertAction(env, dim=0, toggle_at_step=1)
        >>> env = DynamicsHintObservation(env)
        >>> env.observation_space.shape
        (5,)
        >>> state, _ = env.reset(seed=42)
        >>> state
        np.array([-0.14995256, 0.98869318, -0.12224312, 0., 1.])
        >>> state, _, _, _, _ = env.step(np.array([0.5], dtype=np.float32))
        >>> state
        np.array([-0.18417667, 0.98289317, 0.69427675, 1., 0.])
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
    ):
        """A wrapper for adding information on the active action wrapper to the observation.

        Args:
            env: The environment to wrap
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        self._possible_dynamics_wrappers = [
            "InvertAction",
            "NoiseAction",
            "SineNoiseAction",
            "OffsetAction",
            "ScaleAction",
            "RepeatAction",
            "ZeroAction",
            "SwapAction"
        ]
        wrappers = [w.name for w in self.env.get_wrapper_attr("spec").additional_wrappers]
        # + 1 for the default dynamics (no wrapper)
        self.num_dynamics = sum([w in self._possible_dynamics_wrappers for w in wrappers]) + 1

        if isinstance(env.observation_space, Dict):
            new_observation_space = copy.deepcopy(env.observation_space)
            new_observation_space["multi_dynamics_hint"] = Box(
                low=0,
                high=1,
                shape=(self.num_dynamics,),
                dtype=np.int8
            )
        elif isinstance(env.observation_space, Box):
            new_observation_space = Box(
                low=np.concatenate([env.observation_space.low, np.zeros(self.num_dynamics)]),
                high=np.concatenate([env.observation_space.high, np.ones(self.num_dynamics)]),
                shape=(env.observation_space.shape[0] + self.num_dynamics,),
                dtype=env.observation_space.dtype,
            )
        else:
            raise NotImplementedError()
        self.observation_space = new_observation_space
        self._env = env

    def _hint_func(self, observation):
        multi_dynamics_hint = self._get_multi_dynamics_one_hot()
        if isinstance(self.env.observation_space, Dict):
            observation["multi_dynamics_hint"] = multi_dynamics_hint
            return observation
        elif isinstance(self.env.observation_space, Box):
            return np.concatenate([observation, multi_dynamics_hint])
        else:
            raise NotImplementedError()

    def _get_multi_dynamics_one_hot(self):
        multi_dynamics_hint = np.zeros(self.num_dynamics)
        wrapper_looper = self.env
        i_dynamics = 0
        for _ in range(len(self.env.get_wrapper_attr("spec").additional_wrappers)):
            if wrapper_looper.class_name() in self._possible_dynamics_wrappers:
                multi_dynamics_hint[i_dynamics] = wrapper_looper.is_active()
                i_dynamics += 1
            wrapper_looper = wrapper_looper.env
        if multi_dynamics_hint.sum() == 1:
            return multi_dynamics_hint
        elif multi_dynamics_hint.sum() == 0:
            multi_dynamics_hint[i_dynamics] = 1  # default dynamic is active
            return multi_dynamics_hint
        else:
            raise ValueError(
                "When using DynamicsHintObservation, only one dynamics wrapper is supposed to be "
                f"active at the same time. Active are {int(multi_dynamics_hint.sum())}."
            )

    def observation(self, observation: ObsType) -> Any:
        """Apply function to the observation."""
        return self._hint_func(observation)


