# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Pendulum domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import numpy as np


_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))
SUITE = containers.TaggedTasks()


def update_physics(xml_string, mass=1.0, length=0.5, action_factor=1.0):
    """
    Adapts and returns the xml_string of the model with the given context.
    Inspired by
    https://github.com/automl/CARL/blob/main/carl/envs/dmc/dmc_tasks/utils.py
    """

    mjcf = etree.fromstring(xml_string)

    # Update length and mass
    bodies = mjcf.findall("./worldbody/body")
    for body in bodies:
        if body.get("name") == "pole":
            body.set("pos", " ".join(["0", "0", str(length + 0.1)]))
            geoms = body.findall(".//geom")
            for geom in geoms:
                if geom.get("name") == "pole":
                    geom.set("fromto", " ".join(["0", "0", "0", "0", "0", str(length)]))
                elif geom.get("name") == "mass":
                    geom.set("mass", str(mass))
                    geom.set("pos", " ".join(["0", "0", str(length)]))

    xml_string = etree.tostring(mjcf, pretty_print=True)
    return xml_string


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('pendulum.xml'), common.ASSETS


@SUITE.add('benchmarking')
def swingup(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, context_kwargs=None):
    """Returns pendulum swingup task ."""
    xml_string, assets = get_model_and_assets()
    if context_kwargs is not None:
        xml_string = update_physics(xml_string=xml_string, **context_kwargs)
    physics = Physics.from_xml_string(xml_string, assets)
    task = SwingUp(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Pendulum domain."""

    def pole_vertical(self):
        """Returns vertical (z) component of pole frame."""
        return self.named.data.xmat['pole', 'zz']

    def angular_velocity(self):
        """Returns the angular velocity of the pole."""
        return self.named.data.qvel['hinge'].copy()

    def pole_orientation(self):
        """Returns both horizontal and vertical components of pole frame."""
        return self.named.data.xmat['pole', ['zz', 'xz']]


class SwingUp(base.Task):
    """A Pendulum `Task` to swing up and balance the pole."""

    def __init__(self, random=None, action_factor=None):
        """Initialize an instance of `Pendulum`.

        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._action_factor = 1.0 if action_factor is None else action_factor
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Pole is set to a random angle between [-pi, pi).

        Args:
          physics: An instance of `Physics`.

        """
        physics.named.data.qpos['hinge'] = self.random.uniform(-np.pi, np.pi)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation.

        Observations are states concatenating pole orientation and angular velocity
        and pixels from fixed camera.

        Args:
          physics: An instance of `physics`, Pendulum physics.

        Returns:
          A `dict` of observation.
        """
        obs = collections.OrderedDict()
        obs['orientation'] = physics.pole_orientation()
        obs['velocity'] = physics.angular_velocity()
        return obs

    def get_reward(self, physics):
        return rewards.tolerance(physics.pole_vertical(), (_COSINE_BOUND, 1))

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        # Support legacy internal code.
        action = getattr(action, "continuous_actions", action) * self._action_factor
        physics.set_control(action)
