from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np


class ComplexODE(gym.Env):
    """
        Adapted from Beukman et al.
        "Dynamics Generalisation in Reinforcement Learning via Adaptive Context-Aware Policies"
        https://github.com/Michael-Beukman/DecisionAdapter/blob/main/src/genrlise/envs/complex_ode.py

        This is a simple environment that follows a differential equation, parametrised by the context.
        Concretely it is defined as:
        xdot = c0 * a + c1 * a^2 + c2 * a^3 + ..., where ci is the context.

        The difference between this and the above one is that here x dot does not depend on x.
    """
    def __init__(
        self,
        force_mag: float = 1,
        delta_time: float = 0.01,
        max_x: float = 20,
        only_two: bool = True,
        parameter_0: float = 1.0,
        parameter_1: float = 0.0,
        parameter_2: float = 0.0,
        parameter_3: float = 0.0,
        parameter_4: float = 0.0,
        parameter_5: float = 0.0,
        parameter_6: float = 0.0,
        parameter_7: float = 0.0,
        parameter_8: float = 0.0,
        parameter_9: float = 0.0,
        parameter_10: float = 0.0,
        parameter_11: float = 0.0,
        parameter_12: float = 0.0,
        parameter_13: float = 0.0,
        parameter_14: float = 0.0,
        parameter_15: float = 0.0,
        parameter_16: float = 0.0,
        parameter_17: float = 0.0,
        parameter_18: float = 0.0,
        parameter_19: float = 0.0,
    ) -> None:
        """Initialises this environment

        Args:
            force_mag (float):    Maximum force magnitude
            delta_time (float):   Delta Time (x = x + xdot * dt)
            max_x (float):        Bounds on X
            parameter_0 (float):  The first context variable
            parameter_1 (float):  The second context variable
        """

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-max_x, high=max_x, shape=(1,), dtype=np.float32)

        self.x = None

        self.force_mag = force_mag
        self.delta_time = delta_time
        self.max_x = max_x

        if only_two:
            self.parameters = [
                parameter_0,
                parameter_1,
            ]

            self._context = {
                "parameter_0": parameter_0,
                "parameter_1": parameter_1,
            }
            self._context_low = {
                "parameter_0": -np.inf,
                "parameter_1": -np.inf,
            }
            self._context_high = {
                "parameter_0": np.inf,
                "parameter_1": np.inf,
            }
        else:
            self.parameters = [
                parameter_0,
                parameter_1,
                parameter_2,
                parameter_3,
                parameter_4,
                parameter_5,
                parameter_6,
                parameter_7,
                parameter_8,
                parameter_9,
                parameter_10,
                parameter_11,
                parameter_12,
                parameter_13,
                parameter_14,
                parameter_15,
                parameter_16,
                parameter_17,
                parameter_18,
                parameter_19,
            ]

            self._context = {
                "parameter_0": parameter_0,
                "parameter_1": parameter_1,
                "parameter_2": parameter_2,
                "parameter_3": parameter_3,
                "parameter_4": parameter_4,
                "parameter_5": parameter_5,
                "parameter_6": parameter_6,
                "parameter_7": parameter_7,
                "parameter_8": parameter_8,
                "parameter_9": parameter_9,
                "parameter_10": parameter_10,
                "parameter_11": parameter_11,
                "parameter_12": parameter_12,
                "parameter_13": parameter_13,
                "parameter_14": parameter_14,
                "parameter_15": parameter_15,
                "parameter_16": parameter_16,
                "parameter_17": parameter_17,
                "parameter_18": parameter_18,
                "parameter_19": parameter_19,
            }
            self._context_low = {
                "parameter_0": -np.inf,
                "parameter_1": -np.inf,
                "parameter_2": -np.inf,
                "parameter_3": -np.inf,
                "parameter_4": -np.inf,
                "parameter_5": -np.inf,
                "parameter_6": -np.inf,
                "parameter_7": -np.inf,
                "parameter_8": -np.inf,
                "parameter_9": -np.inf,
                "parameter_10": -np.inf,
                "parameter_11": -np.inf,
                "parameter_12": -np.inf,
                "parameter_13": -np.inf,
                "parameter_14": -np.inf,
                "parameter_15": -np.inf,
                "parameter_16": -np.inf,
                "parameter_17": -np.inf,
                "parameter_18": -np.inf,
                "parameter_19": -np.inf,
            }
            self._context_high = {
                "parameter_0": np.inf,
                "parameter_1": np.inf,
                "parameter_2": np.inf,
                "parameter_3": np.inf,
                "parameter_4": np.inf,
                "parameter_5": np.inf,
                "parameter_6": np.inf,
                "parameter_7": np.inf,
                "parameter_8": np.inf,
                "parameter_9": np.inf,
                "parameter_10": np.inf,
                "parameter_11": np.inf,
                "parameter_12": np.inf,
                "parameter_13": np.inf,
                "parameter_14": np.inf,
                "parameter_15": np.inf,
                "parameter_16": np.inf,
                "parameter_17": np.inf,
                "parameter_18": np.inf,
                "parameter_19": np.inf,
            }

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """This steps the simulation one step forward.

        Args:
            action (float): Continuous force between [-1, 1]

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
        """
        assert self.x is not None, "Reset first."
        force = action * self.force_mag
        self._update_ode(force)
        r = self._get_reward(self.x)
        info = {}
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self.x, r, False, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Any:
        super().reset(seed=seed)
        self.x = np.array(self.np_random.random() * 2 - 1).reshape(1, )
        return self.x, {}

    def _get_xdot(self, a) -> float:
        a = complex(a[0], a[1])
        vals = []
        for i, c in enumerate(self.parameters):
            vals.append(a ** (i + 1))
        xdot: complex = sum(c * v for c, v in zip(self.parameters, vals))
        # xdot: complex = self.parameter_0 * a + self.parameter_1 * a**2
        xdot = float(xdot.real)
        return xdot

    def _update_ode(self, a: float):
        """This updates the ODE according to the equation, and given the action

        Args:
            a (float): _description_
        """
        xdot = self._get_xdot(a)
        self.x += xdot * self.delta_time
        self.x = np.clip(self.x, -self.max_x, self.max_x)

    def _get_reward(self, next_state: np.ndarray) -> float:
        s = abs(next_state)
        bounds = [0.05, 0.1, 0.2, 0.5]
        for i, b in enumerate(bounds):
            if s < b:
                return 1. / (i + 1)
        if s < 2:
            return 0.05
        # Reward is either: 1, 0.5, 0.33, 0.25, 0.05 or 0
        return 0.


if __name__ == "__main__":
    env = ComplexODE()
    _, _ = env.reset(seed=0)
    for _ in range(20):
        env.step(env.action_space.sample())

