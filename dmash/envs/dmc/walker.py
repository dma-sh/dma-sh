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

"""Planar Walker Domain."""

import collections
from lxml import etree

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards


_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8


SUITE = containers.TaggedTasks()


def update_physics(xml_string, gravity=9.81, actuator_strength=1.0, leg_length_difference=0.0):
    """
    Adapts and returns the xml_string of the model with the given context.
    Inspired by
    https://github.com/automl/CARL/blob/main/carl/envs/dmc/dmc_tasks/utils.py
    https://github.com/google-research/realworldrl_suite/blob/master/realworldrl_suite/environments/walker.py
    """

    mjcf = etree.fromstring(xml_string)

    # Update gravity
    option = mjcf.find(".//option")
    if option is None:
        option = etree.Element("option")
        mjcf.append(option)
    gravity_default = option.get("gravity")
    if gravity_default is None:
        gravity = " ".join(["0", "0", str(-gravity)])
    else:
        g = gravity_default.split(" ")
        gravity = " ".join([g[0], g[1], str(-gravity)])
    option.set("gravity", gravity)

    # Update actuator strength
    actuators = mjcf.findall("./actuator/")
    for actuator in actuators:
        gear = actuator.get("gear")
        if gear is None:
            gear = 1
        actuator.set("gear", str(float(gear) * actuator_strength))

    # Update leg length
    left_leg_length = 0.25 + (leg_length_difference / 2.0)
    right_leg_length = 0.25 - (leg_length_difference / 2.0)
    geoms = mjcf.findall(".//geom")
    for geom in geoms:
        if geom.get("name") in ["left_leg"]:
            geom.set("size", " ".join(["0.04", str(left_leg_length)]))
        elif geom.get("name") in ["right_leg"]:
            geom.set("size", " ".join(["0.04", str(right_leg_length)]))
    bodies = mjcf.findall(".//body")
    for body in bodies:
        if body.get("name") in ["left_foot"]:
            body.set("pos", " ".join(["0.06", "0", str(-left_leg_length)]))
        elif body.get("name") in ["right_foot"]:
            body.set("pos", " ".join(["0.06", "0", str(-right_leg_length)]))
        elif body.get("name") in ["left_leg"]:
            default_0 = -0.7 + 0.25  # for a leg length 0
            body.set("pos", " ".join(["0", "0", str(default_0 - left_leg_length)]))
        elif body.get("name") in ["right_leg"]:
            default_0 = -0.7 + 0.25  # for a leg length 0
            body.set("pos", " ".join(["0", "0", str(default_0 - right_leg_length)]))
    joints = mjcf.findall(".//joint")
    for joint in joints:
        if joint.get("name") in ["left_knee"]:
            joint.set("pos", " ".join(["0", "0", str(left_leg_length)]))
        elif joint.get("name") in ["right_knee"]:
            joint.set("pos", " ".join(["0", "0", str(right_leg_length)]))

    xml_string = etree.tostring(mjcf, pretty_print=True)
    return xml_string


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('walker.xml'), common.ASSETS


@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, context_kwargs=None):
    """Returns the Stand task."""
    xml_string, assets = get_model_and_assets()
    if context_kwargs is not None:
        xml_string = update_physics(xml_string=xml_string, **context_kwargs)
    physics = Physics.from_xml_string(xml_string, assets)
    task = PlanarWalker(move_speed=0, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, context_kwargs=None):
    """Returns the Walk task."""
    xml_string, assets = get_model_and_assets()
    if context_kwargs is not None:
        xml_string = update_physics(xml_string=xml_string, **context_kwargs)
    physics = Physics.from_xml_string(xml_string, assets)
    task = PlanarWalker(move_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, context_kwargs=None):
    """Returns the Run task."""
    xml_string, assets = get_model_and_assets()
    if context_kwargs is not None:
        xml_string = update_physics(xml_string=xml_string, **context_kwargs)
    physics = Physics.from_xml_string(xml_string, assets)
    task = PlanarWalker(move_speed=_RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker domain."""

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat['torso', 'zz']

    def torso_height(self):
        """Returns the height of the torso."""
        return self.named.data.xpos['torso', 'z']

    def horizontal_velocity(self):
        """Returns the horizontal velocity of the center-of-mass."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]

    def orientations(self):
        """Returns planar orientations of all bodies."""
        return self.named.data.xmat[1:, ['xx', 'xz']].ravel()


class PlanarWalker(base.Task):
    """A planar walker task."""

    def __init__(self, move_speed, random=None):
        """Initializes an instance of `PlanarWalker`.

        Args:
          move_speed: A float. If this value is zero, reward is given simply for
            standing up. Otherwise this specifies a target horizontal velocity for
            the walking task.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._move_speed = move_speed
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        In 'standing' mode, use initial orientation and small velocities.
        In 'random' mode, randomize joint angles and let fall to the floor.

        Args:
          physics: An instance of `Physics`.

        """
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of body orientations, height and velocites."""
        obs = collections.OrderedDict()
        obs['orientations'] = physics.orientations()
        obs['height'] = physics.torso_height()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(physics.torso_height(),
                                     bounds=(_STAND_HEIGHT, float('inf')),
                                     margin=_STAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4
        if self._move_speed == 0:
            return stand_reward
        else:
            move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                            bounds=(self._move_speed, float('inf')),
                                            margin=self._move_speed / 2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
            return stand_reward * (5 * move_reward + 1) / 6
