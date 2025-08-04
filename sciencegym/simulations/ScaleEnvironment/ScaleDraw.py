#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version Copyright (c) 2010 kne / sirkne at gmail dot com
#
# Implemented using the pybox2d SWIG interface for Box2D (pybox2d.googlecode.com)
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

import math
import sys

from time import time, sleep

import numpy as np
from collections import defaultdict

import Box2D  # The main library
from Box2D import b2Color, b2Vec2, b2DrawExtended
from gym import spaces
from gym.spaces import Discrete, Dict, Box
from gym.utils import seeding
from pyglet.math import Vec2

# from ScaleEnvironment.framework import (Framework, Keys, main)
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, edgeShape, fixtureDef)

import gym
import pygame
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 20.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

colors = {
    staticBody: (255, 255, 255, 255),
    dynamicBody: (127, 127, 127, 255),
}

groundColor = (255, 255, 255)
triangleColor = (255, 255, 255)
barColor = (255, 0, 0)

# --- constants ---
BOXSIZE = 1.0
DENSITY = 5.0
BARLENGTH = 15

FAULTTOLERANCE = 0.001  # for the angle of the bar
ANGLE_TRESHOLD = 0.98

WAITINGITERATIONS = 20  # maximum iterations to wait per episode
MAXITERATIONS = 1000


def rescale_movement(original_interval, value, to_interval=(-BARLENGTH, +BARLENGTH)):
    """
    Help function to do and to undo the normalization of the action and observation space

    :param original_interval: Original interval, in which we observe the value.
    :type original_interval: list[float, float]
    :param value: Number that should be rescaled.
    :type value: float
    :param to_interval: New interval, in which we want to rescale the value.
    :type to_interval: list[float, float]
    :return: Rescaled value
    :rtype: float
    """
    a, b = original_interval
    c, d = to_interval
    return c + ((d - c) / (b - a)) * (value - a)


class ScaleDraw(gym.Env):
    name = "Scale"  # Name of the class to display

    def __init__(self, rendering=True, random_densities=True, random_boxsizes=False, normalize=False, placed=1,
                 actions=1, sides=2, raw_pixels=False):
        """
        Initialization of the Scale Environment

        :param rendering: Should the experiment be rendered or not
        :type rendering: bool
        :param random_densities: if True: randomized densities (from 4.0 to 6.0), else: fixed density which is set to 5.0
        :type random_densities: bool
        :param random_boxsizes: if True: randomzied sizes of the box (from 0.8 to 1.2), else: fixed box size which is set to 1.0
        :type random_boxsizes: bool
        :param normalize: Should the state and actions be normalized to values between 0 to 1 (or for the positions: -1 to 1) for the agent?
        :type normalize: bool
        :param placed: How many boxes should be placed randomly individually?
        :type placed: int
        :param actions: How many boxes should the agent place on the scale?
        :type actions: int
        :param sides: if 1: divided into 2 sides, placed boxes on the left and the agent to place by the agent on the right side; if 2: boxes can be dropped anywhere on the bar (except for the edges)
        :type sides: int
        :param raw_pixels: if True: the agent gets an pixel array as input, else: agent gets the observation space as an accumulation of values (positions, densities, boxsizes, bar angle, velocitiy of the bar, ...)
        :type raw_pixels: bool
        """

        self.np_random = None
        self.seed()

        self.num_envs = 1  # for stable-baseline3

        # Initialize all the objects
        self.y = 6.0 + BOXSIZE
        BARHEIGHT = 6  # height of the bar joint

        # screen / observation space measurements
        self.height = SCREEN_HEIGHT  # 480
        self.width = SCREEN_WIDTH  # 640
        factor = 1
        self.height //= factor
        self.width //= factor

        # Pygame setup
        if rendering:
            pygame.init()

            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('Box Gym')
        else:
            self.screen = pygame.display.set_mode((1, 1))
        self.clock = pygame.time.Clock()

        # Box2d world setup
        # Create the world
        self.world = world(gravity=(0, -9.80665), doSleep=True)

        self.counter = 0
        self.timesteps = 0
        self.reward = 0

        self.rendering = rendering  # should the simulation be rendered or not
        self.random_densities = random_densities  # random densities or are both the same
        self.random_boxsizes = random_boxsizes
        self.normalize = normalize
        self.placed = placed
        self.actions = actions
        self.sides = sides
        if self.actions <= 0 or self.placed < 0:
            assert ValueError("Should be one or more actions and a positive number of placed boxes")
        if self.sides not in {1, 2}:
            assert ValueError("Sides value should be either 1 or 2")
        self.raw_pixels = raw_pixels

        self.order = None

        # action space determination
        limit1, limit2 = BARLENGTH - 2 * BOXSIZE, 2 * BOXSIZE
        if not self.normalize:
            self.action_space = Box(
                low=np.array([-limit1 if not self.sides == 1 else limit2 for _ in range(actions)]),
                high=np.array([limit1 for _ in range(actions)]),
                shape=(self.actions,), dtype=np.float32)
        else:
            self.action_space = Box(low=np.array([-1 if self.sides == 2 else 0 for _ in range(actions)]),
                                    high=np.array([1 for _ in range(actions)]),
                                    shape=(self.actions,), dtype=np.float32)

        # observation space
        if not raw_pixels:
            observation_dict = {
                "angle": Box(low=-0.390258252620697, high=0.390258252620697, shape=(1,), dtype=float),
                # angular velocity of the bar, negative: moves to the right, positive: moves to the left
                "vel": Box(low=-2., high=2., shape=(1,), dtype=float),
            } if not self.normalize else {
                "angle": Box(low=-1, high=1., shape=(1,), dtype=float),
                "vel": Box(low=-1., high=1., shape=(1,), dtype=float),
            }

            for i in range(1, self.placed + self.actions + 1):
                observation_dict[f"position{i}"] = Box(low=-1. if self.normalize else -20.,
                                                       high=1. if self.normalize else 20., shape=(1,), dtype=float)
                observation_dict[f"density{i}"] = Box(low=0. if self.normalize else 4.,
                                                      high=1. if self.normalize else 6., shape=(1,), dtype=float)
                observation_dict[f"boxsize{i}"] = Box(low=0. if self.normalize else 0.8,
                                                      high=1. if self.normalize else 1.2, shape=(1,), dtype=float)

            self.observation_space = spaces.Dict(spaces=observation_dict)  # convert to Spaces Dict

        else:
            """self.observation_space = spaces.Box(low=0, high=1. if self.normalize else 255,
                                                shape=(self.width, self.height, 3),
                                                dtype=np.float32 if self.normalize else np.uint8)"""
            dummy_obs = self.render("rgb_array")
            self.observation_space = spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

        # setting up the objects on the screen
        self.ground = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=polygonShape(box=(40, 1)),
            userData=groundColor,
        )

        self.maxAngle = 0.390258252620697  # self.getMaxAngle() # todo: fix getMaxAngle function

        topCoordinate = Vec2(0, BARHEIGHT)
        self.triangle = self.world.CreateStaticBody(
            position=(0, 0),
            fixtures=fixtureDef(shape=polygonShape(vertices=[(-1, 0), (1, 0), topCoordinate]), density=100),
            userData=triangleColor,
        )

        self.bar = self.world.CreateDynamicBody(
            position=topCoordinate,
            fixtures=fixtureDef(shape=polygonShape(box=(BARLENGTH, 0.3)), density=1),
            userData=barColor,
        )

        # connect the bar with the triangle
        self.joint = self.world.CreateRevoluteJoint(bodyA=self.bar, bodyB=self.triangle, anchor=topCoordinate)

        # reset every dict/array and all the boxes on the screen
        self.boxes = defaultdict(lambda: None)
        self.boxsizes = defaultdict(lambda: None)
        self.densities = defaultdict(lambda: None)
        self.positions = defaultdict(lambda: None)
        self.joints = defaultdict(lambda: None)
        self.reset()

        # state calculation
        self.state = None
        self.internal_state = None
        self.state = self.resetState()

        self.normalized_state = None
        return

    def convertDensityToRGB(self, density, low=4., high=6., channels=[True, True, True]):
        """
        Gets a value for the density of one box and returns the corresponding color

        :param density: density of the box (should be in range of the interval)
        :type density: float
        :param low: the minimum value for the d ensity
        :type low: float
        :param high: the maximum value for the density
        :type high: float
        :param channels: an array with 3 entries, where each entry says if the color channel should be used or not.
        e.g. if it's [True, False, True], we want to use the Red and Blue channel, but not the Green channel.
        :type channels: list[bool, bool, bool]
        :return: a RGB color
        :rtype: (int, int, int)
        """
        if not (low <= density <= high):
            raise AssertionError(f"Density {density} not in allowed range [{low},{high}]")

        if len(channels) != 3 or not all(type(channel) == bool for channel in channels):
            raise TypeError("Type of the channels array has to be a List of 3 Bool values.")

        total_number_of_colors = 256 ** sum(channels)
        # first normalize the density
        density = int(rescale_movement([low, high], density, [0., total_number_of_colors - 1]))

        RGB = [0, 0, 0]

        index = 0
        for i in reversed([i for i, boolean in enumerate(channels) if boolean]):
            RGB[i] = (density >> (index * 8)) & 255
            index += 1
        red, green, blue = RGB
        return red, green, blue

    def convertRGBToDensity(self, RGB, low=4., high=6., channels=[True, True, True]):
        """
        Gets a RGB value of an box and returns the corresponding density of the box

        :param RGB: (red, green, blue) values
        :type RGB: (int, int, int)
        :param low: the minimum value for the density
        :type low: float
        :param high: the maximum value for the density
        :type high: float
        :return: density value
        :rtype: float
        """
        if not all(0 <= colorVal <= 255 for colorVal in RGB):
            raise AssertionError(f"RGB value {RGB} not allowed!")

        if len(channels) != 3 or (type(channels[i]) != bool for i in range(3)):
            raise TypeError("Type of the channels array has to be a List of 3 Bool values.")

        total_number_of_colors = 256 ** sum(channels)

        value = 0
        index = 0

        for i in reversed([i for i, boolean in enumerate(channels) if boolean]):
            value += RGB[i] * 256 ** index
            index += 1

        value /= total_number_of_colors

        # rescale the density
        density = rescale_movement([0., total_number_of_colors - 1], value, [low, high])

        return density

    def convertDensityToGrayscale(self, density, low=4., high=6.):  # todo: fix
        """
        Gets a value for the density of one box and returns the corresponding grayscale value

        :param density: density of the box (should be in range of the interval)
        :type density: float
        :param low: the minimum value for the density
        :type low: float
        :param high: the maximum value for the density
        :type high: float
        :return: a RGB color
        :rtype: (int, int, int)
        """
        colormap = cm.gray
        norm = Normalize(vmin=low, vmax=high)
        red, green, blue, brightness = colormap(norm(density))

        return red, green, blue  # , brightness

    def convertGrayscaleToDensity(self, RGB, low=4., high=6.):  # todo
        """
        Gets a Grayscale value of an box and returns the corresponding density of the box

        :param RGB: (red, green, blue) values
        :type RGB: (int, int, int)
        :param low: the minimum value for the density
        :type low: float
        :param high: the maximum value for the density
        :type high: float
        :return: density value
        :rtype: float
        """
        density = None
        return density

    def getMaxAngle(self):  # todo: fix
        """
        Only called at start to calculate the biggest possible angle.
        Use this to have a good termination condition for the learning process.

        :return: the value of the maximum possible angle
        :rtype: float
        """
        self.maxAngle = math.pi / 2
        self.placeBox(self.boxes[1], - BARLENGTH + 2)
        self.placeBox(self.boxes[2], BARLENGTH + 3)
        self.internal_step(None)
        while self.bar.angularVelocity != 0:
            self.internal_step(None)
        angle = self.bar.angle
        self.reset()
        return abs(angle)

    def createBox(self, pos_x, pos_y=None, density=DENSITY, boxsize=BOXSIZE, index=0):
        """
        Create a new box on the screen

        :param pos_x: x position of the box (in pixel coordinates, not as a measurement of the distance to the center of the bar!)
        :param pos_y: y position of the box, if None: use default position
        :param density: density of the box, if not given: use default density = 5.0
        :param boxsize: size of the box, if not given: use default boxsize = 1.0
        :param index: index of the box for the dictionaries, if not given: calculated inside of the function
        :return: new (dynamic) box
        """
        try:
            pos_x = float(pos_x[0])
            pos_y = float(pos_y[0])
        except:
            pass
        if not pos_y:
            pos_y = self.y
        newBox = self.world.CreateDynamicBody(
            position=(pos_x, pos_y),
            fixtures=fixtureDef(shape=polygonShape(box=(boxsize, boxsize)),
                                density=density, friction=100.),
            # userData=self.convertDensityToGrayscale(density=density, low=4., high=6.)
        )

        if index == 0:
            index = len(self.boxes.values()) + 1
        self.boxes[index] = newBox
        self.boxsizes[index] = boxsize
        return newBox

    def deleteBox(self, box):
        """
        Delete a box from the world and from the screen.

        :param box: box object which we want to delete
        :type box: Box2D.b2Body
        """
        if box not in self.boxes.values():
            print("Box not found")
            return
        key = list(self.boxes.keys())[list(self.boxes.values()).index(box)]
        del self.boxes[key]
        self.world.DestroyBody(box)
        return

    def deleteAllBoxes(self):
        """
        Deletes every single box in the world
        """
        """for joint in self.joints.values():
            self.world.DestroyJoint(joint)
        self.joints = defaultdict(lambda: None)"""

        for box in self.boxes.values():
            try:
                self.world.DestroyBody(box)
            except Exception as e:
                print(e)
        self.boxsizes = defaultdict(lambda: None)
        self.boxes = defaultdict(lambda: None)
        return

    def resetBoxes(self):
        """
        Generate the boxes and place all of the ones that are supposed to be placed randomly

        :return: self.boxes, a dictionary with every box as value and its index as key
        :rtype: Dict[int, Box2D.b2Body]
        """

        def overlapping(boxes):
            """
            Help function to determine whether two boxes would overlap when trying to place them to their
            starting positions given their sizes.

            :param boxes: the boxes we want to check for overlapping (at least 2)
            :return: True if any boxes collide, if not: False
            """
            for (i, (box_position1, box_size1)) in enumerate(boxes):
                for (box_position2, box_size2) in boxes[i + 1:]:
                    # check if the ranges of the boxes overlap
                    if max(0, min(box_position1 + box_size1, box_position2 + box_size2) -
                              max(box_position1 - box_size1, box_position2 - box_size2)) > 0:
                        return True
            return False

        # choose positions and sizes for boxes
        boxes = []
        while (len(boxes) < self.placed):
            if self.sides == 1:
                position = self.np_random.uniform(- BARLENGTH + 2 * BOXSIZE, -2 * BOXSIZE)
            else:
                position = self.np_random.uniform(- BARLENGTH + 2 * BOXSIZE, BARLENGTH - 2 * BOXSIZE)
            boxsize = self.np_random.uniform(0.8, 1.2) if self.random_boxsizes else BOXSIZE
            if not overlapping(boxes + [(position, boxsize)]):
                boxes.append((position, boxsize))

        # save values of densities and box sizes and place the boxes
        self.boxes = defaultdict(lambda: None)
        self.boxsizes = defaultdict(lambda: None)
        self.densities = defaultdict(lambda: None)
        self.positions = defaultdict(lambda: None)
        i = 0
        for (i, (pos, size)) in enumerate(boxes, start=1):
            density = 4. + 2 * self.np_random.random()  # between 4 and 6
            box = self.createBox(pos_x=pos, pos_y=size,
                                 density=density if self.random_densities else DENSITY,
                                 boxsize=size if self.random_boxsizes else BOXSIZE,
                                 index=i)
            self.boxes[i] = box
            self.densities[i] = density if self.random_densities else DENSITY
            self.boxsizes[i] = size if self.random_boxsizes else BOXSIZE
            self.positions[i] = pos

        # generate the boxes that should be placed randomly
        for j in range(self.actions):
            index = i + j + 1
            position = - BARLENGTH - 2 * j * BOXSIZE
            boxsize = self.np_random.uniform(0.8, 1.2) if self.random_boxsizes else BOXSIZE
            density = 4. + 2 * self.np_random.random()
            box = self.createBox(pos_x=position, pos_y=boxsize,
                                 density=density if self.random_densities else DENSITY,
                                 boxsize=boxsize if self.random_boxsizes else BOXSIZE,
                                 index=index)
            self.boxes[index] = box
            self.densities[index] = density if self.random_densities else DENSITY
            self.boxsizes[index] = boxsize if self.random_boxsizes else BOXSIZE
            self.positions[index] = position
        return self.boxes

    def moveBox(self, box, deltaX, deltaY, index=0):
        """
        Move a box in the world along a given vector (deltaX,deltaY)

        :param box: box to be moved
        :type box: Box2d.b2Body
        :param deltaX: How much do we want the box to be moved to the left or right?
        :type deltaX: float
        :param deltaY: How much do we want the box to be moved up or down?
        :type deltaY: float
        :param index: need to pass this, because we create a duplicate of the given box and delete the old one
        :type index: int
        :return: the same box with the new positions
        :rtype: Box2d.b2Body
        """
        x = box.position[0] + deltaX
        y = box.position[1] + deltaY
        return self.moveBoxTo(box, x, y, index)

    def moveBoxTo(self, box, x, y, index=0):
        """
        Place a box at a specific position on the field

        :param box: Which box should be moved?
        :type box: Box2d.b2Body
        :param x: new x position of the box
        :type x: float
        :param y: new x position of the box
        :type y: float
        :param index: index of the box
        :type index: int
        :return: the moved box
        :rtype: Box2d.b2Body
        """
        self.deleteBox(box)
        boxsize = self.boxsizes[index]
        density = self.densities[index]
        movedBox = self.createBox(x, y, density, boxsize, index=index)
        return movedBox

    def placeBox(self, box, pos, index=0):
        """
        Place a box on the scale

        :param box:
        :param pos:
        :param index:
        :return:
        """
        x = math.cos(self.bar.angle) * pos
        y = 6 + math.tan(self.bar.angle) * pos + BOXSIZE
        placedBox = self.moveBoxTo(box, x, y, index=index)
        placedBox.angle = self.bar.angle

        # self.joints[f"{index}L"] = self.world.CreateRevoluteJoint(bodyA=self.bar, bodyB=placedBox, anchor=b2Vec2(x - 0.5, 6.3))
        # self.joints[f"{index}R"] = self.world.CreateRevoluteJoint(bodyA=self.bar, bodyB=placedBox, anchor=b2Vec2(x + 0.5, 6.3))
        return placedBox

    def moveAlongBar(self, box, delta_pos, index=0):
        """
        Take a box and move it along the bar with the given distance

        :param box: box to be moved along the bar
        :type box: Box2d.body
        :param delta_pos: how much should the box be moved (negative value --> left, positive value --> right)
        :type delta_pos: float
        :param index: index of the box
        :type index: int
        :return: the moved box
        :rtype: Box2d.b2Body
        """
        # recalculate the position on the bar
        pos = box.position[0] / math.cos(self.bar.angle)
        pos += delta_pos
        return self.placeBox(box, pos, index=index)

    def resetState(self):
        """
        Resets and returns the current values of the state

        :return: the new state
        :rtype: np.ndarray
        """
        positions = []
        densities = []
        boxsizes = []
        for i in range(1, self.placed + self.actions + 1):
            box = self.boxes[i]
            pos = box.position[0] / math.cos(self.bar.angle)
            positions.append(pos)
            densities.append(self.densities[i])
            boxsizes.append(self.boxsizes[i])
        self.state = np.array(positions + [self.bar.angle, self.bar.angularVelocity] + densities + boxsizes,
                              dtype=np.float32)
        self.internal_state = self.state.copy()

        if self.raw_pixels:
            # self.state = self.render("state_pixels")
            self.state = self.render(mode='rgb_array')  # (self.screen, (self.width, self.height))

        if self.normalize:
            self.normalized_state = self.rescaleState()
        return self.state

    def rescaleState(self, state=None):
        """
        Returns normalized version of the state

        :param state: the normal state, which is not normalized yet
        :type state: np.ndarray
        :return: the new normalized state
        :rtype: np.ndarray
        """
        if state is None:
            state = self.state

        if self.raw_pixels:
            normalized_state = rescale_movement([0, 255], self.state, [0., 1.])

        else:
            n = self.actions + self.placed
            positions: list[float] = rescale_movement([-20., 20.], self.state[0:n], [-1., 1.])
            angle: float = rescale_movement([-self.maxAngle, self.maxAngle], state[2], [-1, 1])
            angularVelocity: float = rescale_movement([-2., 2.], state[3], [-1, 1])
            densities: list[float] = rescale_movement([0., 6.], self.state[n + 2:2 * n + 2], [0., 1.])
            boxsizes: list[float] = rescale_movement([0.8, 1.2], self.state[2 * n + 2:3 * n + 2], [0., 1.])
            normalized_state = np.concatenate((positions, [angle], [angularVelocity], densities, boxsizes))
        return normalized_state

    def performActions(self, actions):
        """
        Get the new positions of the boxes and translate it into actions.

        :param actions: new positions of each box that can be moved by the agent
        :return:
        """
        # transform normalized inputs into world coordinates
        if self.normalize:
            if self.sides == 1:
                actions = rescale_movement([0., 1.], actions, [2 * BOXSIZE, BARLENGTH - 2 * BOXSIZE])
            else:
                actions = rescale_movement([-1., 1.], actions, [-BARLENGTH + 2 * BOXSIZE, BARLENGTH - 2 * BOXSIZE])

        i = 0
        for i in range(1, self.placed + 1):
            self.boxes[i] = self.placeBox(self.boxes[i], self.boxes[i].position[0], index=i)

        for j, action in enumerate(actions):
            self.boxes[i + j + 1] = self.placeBox(self.boxes[i + j + 1], action, index=i + j + 1)
        return

    def step(self, action):
        """
        Actual step function called by the agent

        :param action: the action(s) the agent chooses as an array of each new positions for each box that should be moved
        :type action: list[float]
        :return: new state after step, the reward, done, info
        :rtype: tuple[np.ndarray, float, bool, dict]
        """
        # catch special case that one action is passed as a single input instead of an array
        if type(action) in {float, np.float32, np.float64}:
            action = np.array([action])

        if len(action) != self.actions:
            raise AssertionError(
                f"Number of values in array {len(action)} does not match number of actions {self.actions}!")

        # print(action, self.state.shape)
        timesteps = 120
        # self.action = action
        done = False
        for _ in range(timesteps):
            self.old_state = self.state
            self.state, reward, done, info = self.internal_step(action)
            action = None
            if done:
                break
        if not done:
            done = True
            self.reset()
        return self.state, reward, done, info

    def internal_step(self, action=None):
        """
        Simulates the program with the given action and returns the observations

        :param action: the action(s) the agent chooses as an array of each new positions for each box that should be moved
        :type action: list[float]
        :return: new state after step, the reward, done, info
        :rtype: tuple[np.ndarray, float, bool, dict]
        """

        def boxesOnScale():  # todo: need to change this function for multiple boxes (probably)
            """
            Utility function to check if both boxes are still on the scale

            :return: True, if all(!) boxes are on the scale, else: False
            :rtype: bool
            """
            for box in list(self.boxes.values()):
                if len(box.contacts) < 1:
                    return False
            val = len(self.bar.contacts) == self.actions + self.placed
            return True  # val

        def getReward():
            """
            Calculates the reward and adds it to the self.reward value

            :return: the total reward
            :rtype: float
            """
            # Calculate reward (Scale in balance?)
            if boxesOnScale():
                # both boxes on one side: negative reward
                if abs(self.bar.angle) < FAULTTOLERANCE and boxesOnScale():
                    reward = 1
                else:
                    reward = (self.maxAngle - abs(self.bar.angle)) / self.maxAngle
                self.reward += reward
            else:  # one or both boxes not on the scale
                reward = - 1
            return reward

        # Don't do anything if the setting's Hz are <= 0
        velocityIterations = 8
        positionIterations = 3
        velocityIterations *= 1
        positionIterations *= 1

        self.counter += 1
        self.timesteps += 1

        # check if test failed --> return reward
        if (abs(self.bar.angle) > ANGLE_TRESHOLD * self.maxAngle
                or self.timesteps > MAXITERATIONS):
            state = self.resetState().copy()
            reward = getReward()
            self.render()
            # self.render(mode="state_pixels" if self.raw_pixels else "human")
            self.reset()
            if self.normalize:
                return self.rescaleState(state), reward, True, {}
            return state, reward, True, {}

        # check if no movement anymore
        if self.bar.angularVelocity == 0 and boxesOnScale() and abs(self.bar.angle) < 0.05:
            # check if time's up
            if self.counter > WAITINGITERATIONS:
                # self.render()
                state = self.resetState()
                tab = '\t'
                print(
                    f"Match: [{tab.join([str(box.position[0] / math.cos(self.bar.angle)) for box in self.boxes.values()])}]\t{self.bar.angle}\t{20 * math.cos(self.bar.angle)}")
                reward = getReward()
                # print(self.action)
                self.reset()
                if self.normalize:
                    return self.rescaleState(state), 20 * math.cos(self.bar.angle), True, {}
                return state, 20 * math.cos(self.bar.angle), True, {}
                # return state, 2 * MAXITERATIONS * math.cos(self.bar.angle), True, {}
        else:  # no movement --> reset counter
            self.counter = 0

        # catch special case that no action was executed
        if action is None:
            self.world.Step(TIME_STEP, velocityIterations,
                            positionIterations)
            self.world.ClearForces()
            # self.render(mode="state_pixels" if self.raw_pixels else "human")
            self.render()
            getReward()
            reward = getReward()
            # reward = self.timesteps
            self.state = self.resetState()
            if self.normalize:
                return self.rescaleState(), reward, False, {}
            return self.state, reward, False, {}

        # Place the boxes
        self.performActions(actions=action)

        # Reset the collision points
        self.points = []

        # Tell Box2D to step
        self.world.Step(TIME_STEP, velocityIterations, positionIterations)
        self.world.ClearForces()

        self.description = f"{self.joint.angle * 180 / math.pi}°"

        self.state = self.resetState()

        # Calculate reward (Scale in balance?)
        reward = getReward()

        # no movement and in balance --> done
        done = False

        # placeholder for info
        info = {}

        self.render()
        # self.render(mode="state_pixels" if self.raw_pixels else "human")

        if self.normalize:
            return self.rescaleState(), reward, done, info
        return self.state, reward, done, info

    def close(self):
        """Close the pygame window and terminate the program."""
        pygame.quit()
        sys.exit()

    def render(self, mode="human"):
        """
        Render function, which runs the simulation and render it (if wished)

        :param mode: "human" for rendering, "state_pixels" for returning the pixel array
        :type mode: str
        :return: nothing if mode is human, if mode is "state_pixels", we return the array of the screen
        :rtype: np.array
        """
        assert mode in ["human", "rgb_array", "state_pixels"], f"Wrong render mode passed, {mode} is invalid."

        # Draw Functions
        def my_draw_polygon(polygon, body, fixture):
            vertices = [(body.transform * v) * PPM for v in polygon.vertices]
            vertices = [(v[0] + SCREEN_WIDTH / 2, SCREEN_HEIGHT - v[1]) for v in vertices]
            if body.userData is not None:
                pygame.draw.polygon(self.screen, body.userData, vertices)
            else:
                # pygame.draw.polygon(self.screen, self.convertDensityToRGB(density=fixture.density), vertices)
                # here: don't use the red color channel, only use green and blue
                pygame.draw.polygon(self.screen,
                                    self.convertDensityToRGB(density=fixture.density, low=3.5, high=6.5, channels=[False, True, True]),
                                    vertices)

            """if body.userData is not None:
                pygame.draw.polygon(self.screen, body.userData, vertices)
            else:
                if 4 <= fixture.density <= 6:
                    pygame.draw.polygon(self.screen, self.convertDensityToGrayscale(fixture.density, 3, 7), vertices)
                else:
                    pygame.draw.polygon(self.screen, colors[body.type], vertices)"""

        polygonShape.draw = my_draw_polygon

        def my_draw_circle(circle, body, fixture):
            position = body.transform * circle.pos * PPM
            position = (position[0], SCREEN_HEIGHT - position[1])
            pygame.draw.circle(self.screen, colors[body.type], [int(
                x) for x in position], int(circle.radius * PPM))

        circleShape.draw = my_draw_circle

        def my_draw_edge():
            pass
            # todo: write the function (if necessary)

        edgeShape.draw = my_draw_edge()

        if mode == "rgb_array":
            return self._create_image_array(self.screen, (self.width, self.height))

        elif mode == "state_pixels":
            return self._create_image_array(self.screen, (self.width, self.height))

        elif mode == "human":
            if self.rendering:
                try:
                    self.screen.fill((0, 0, 0, 0))
                except:
                    return
                # Draw the world
                for body in self.world.bodies:
                    for fixture in body.fixtures:
                        fixture.shape.draw(body, fixture)

                # Make Box2D simulate the physics of our world for one step.
                # self.world.Step(TIME_STEP, 10, 10)

                pygame.display.flip()
            # self.clock.tick(TARGET_FPS)
            return None

    def _create_image_array(self, screen, size):
        """
        Use the pygame framework to calculate the 3d pixels array

        :param screen: self.screen
        :type screen: pygame.Surface
        :param size: (height, width) of the screen, use self.height and self.width for our setting
        :type size: tuple[int, int]
        :return: 3d pixels array
        :rtype: np.array
        """
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.array(pygame.surfarray.pixels3d(scaled_screen))

    def reset(self):
        """
        Reset function for the whole environment. Inside of it, we reset every counter/reward, we reset the boxes, the state and all other necessary things.

        :return: the new state (normalized, if wished)
        :rtype: np.ndarray
        """
        # rearrange the bar to 0 degree
        self.bar.angle = 0
        self.bar.angularVelocity = 0.

        self.deleteAllBoxes()

        self.resetBoxes()

        # Reset the reward and the counters
        self.counter = 0
        self.timesteps = 0
        self.reward = 0

        # return the observation
        self.resetState()
        return self.rescaleState() if self.normalize else self.state

    def seed(self, seed=None):
        """
        Seed function for the random number calculation

        :param seed: if None: cannot recreate the same results afterwards
        :type seed: int
        :return: the seed
        :rtype: list[int]
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
