"""
Taken and modified from Shimmy
https://github.com/Farama-Foundation/Shimmy/blob/main/shimmy/registration.py

Registers environments within gymnasium for optional modules.
"""

from __future__ import annotations

import os
if os.uname()[0] == "Linux":
    os.environ["MUJOCO_GL"] = "egl"

from functools import partial
from typing import Any

from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation
import inspect

# Import all domains/tasks.
from dmash.envs.dmc import walker, ball_in_cup, finger, cartpole, pendulum, quadruped, humanoid

DM_CONTROL_SUITE_ENVS = (
    ("walker", "walk"),
    ("ball_in_cup", "catch"),
    ("finger", "spin"),
    ("finger", "turn_easy"),
    ("cartpole", "balance"),
    ("cartpole", "balance_sparse"),
    ("pendulum", "swingup"),
    ("quadruped", "walk"),
    ("humanoid", "stand"),
    ("humanoid", "walk"),
)

# Find all domains imported.
_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}


def load(
    domain_name,
    task_name,
    task_kwargs=None,
    environment_kwargs=None,
    context_kwargs=None,
    visualize_reward=False
):
    """Returns an environment from a domain name, task name and optional settings.

    ```python
    env = suite.load('cartpole', 'balance')
    ```

    Args:
      domain_name: A string containing the name of a domain.
      task_name: A string containing the name of a task.
      task_kwargs: Optional `dict` of keyword arguments for the task.
      environment_kwargs: Optional `dict` specifying keyword arguments for the
        environment.
      visualize_reward: Optional `bool`. If `True`, object colours in rendered
        frames are set to indicate the reward at each step. Default `False`.

    Returns:
      The requested environment.
    """
    if domain_name not in _DOMAINS:
        raise ValueError('Domain {!r} does not exist.'.format(domain_name))

    domain = _DOMAINS[domain_name]

    if task_name not in domain.SUITE:
        raise ValueError('Level {!r} does not exist in domain {!r}.'.format(
            task_name, domain_name))

    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = dict(task_kwargs, environment_kwargs=environment_kwargs)
    if context_kwargs is not None:
        task_kwargs = dict(task_kwargs, context_kwargs=context_kwargs)
    env = domain.SUITE[task_name](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env


def _register_dm_control_envs():
    """Registers all dm-control environments in gymnasium."""
    from dmash.envs.dmc.compatibility import DmControlCompatibilityV0

    def _make_dm_control_suite_env(
        domain_name: str,
        task_name: str,
        task_kwargs: dict[str, Any] | None = None,
        environment_kwargs: dict[str, Any] | None = None,
        visualize_reward: bool = False,
        render_mode: str | None = None,
        render_kwargs: dict[str, Any] | None = None,
        **context_kwargs,
    ):
        """The entry_point function for registration of dm-control environments."""
        env = load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            context_kwargs=context_kwargs,
            visualize_reward=visualize_reward,
        )
        # Flatten because dmc is usually dict env, although not goal-conditioned
        return FlattenObservation(
            DmControlCompatibilityV0(env, render_mode, render_kwargs, context_kwargs)
        )

    for _domain_name, _task_name in DM_CONTROL_SUITE_ENVS:
        register(
            f"dm_control/{_domain_name}-{_task_name}-v0",
            partial(
                _make_dm_control_suite_env,
                domain_name=_domain_name,
                task_name=_task_name,
            ),
        )


def register_dmc_envs():
    _register_dm_control_envs()
