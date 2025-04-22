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
import random
import sys
from time import time, sleep

import numpy as np
import pygame

from Box2D import b2Color, b2Vec2, b2DrawExtended
from gym import spaces
from gym.spaces import Discrete, Dict, Box
from gym.utils import seeding
from pyglet.math import Vec2

from ScaleEnvironment.framework import (Framework, Keys, main)
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape)

import gym
import torch

BOXSIZE = 1.0
DENSITY = 5.0
BARLENGTH = 15  # 18

FAULTTOLERANCE = 0.001  # for the angle of the bar
ANGLE_TRESHOLD = 0.98

WAITINGITERATIONS = 20  # maximum iterations to wait per episode
MAXITERATIONS = 1000


def rescale_movement(original_interval, value, to_interval=(-BARLENGTH, +BARLENGTH)):
    a, b = original_interval
    c, d = to_interval
    return c + ((d - c) / (b - a)) * (value - a)


class ScaleExperiment(Framework, gym.Env):
    """You can use this class as an outline for your tests."""
    name = "ScaleExperiment"  # Name of the class to display

    def __init__(self, rendering=True, random_densities=True, random_boxsizes=False, normalize=False, actions=1, boxes=2):
        """
        Initialize all of your objects here.
        Be sure to call the Framework's initializer first.
        """
        super(ScaleExperiment, self).__init__(rendering)

        self.np_random = None
        self.seed()

        self.num_envs = 1  # for stable-baseline3

        # Initialize all of the objects
        self.y, L, a, b = 6.0 + BOXSIZE, 12.0, 1.0, 2.0
        HEIGHT = 6  # height of the bar joint

        self.counter = 0
        self.timesteps = 0
        self.reward = 0

        self.rendering = rendering  # should the simulation be rendered or not
        self.random_densities = random_densities  # random densities or are both the same
        self.random_boxsizes = random_boxsizes
        self.normalize = normalize
        self.actions = actions  # 1: agent chooses one position, 2 or 3: agent chooses both/all three positions
        self.number_of_boxes = boxes

        # action space determination
        limit1, limit2 = BARLENGTH - 2 * BOXSIZE, 2 * BOXSIZE
        if self.actions == 1:  # only choose to place the right box
            if not self.normalize:
                self.action_space = gym.spaces.Box(low=limit2, high=limit1,  # todo: normalize it
                                                   shape=(1,), dtype=np.float32)
            else:
                self.action_space = gym.spaces.Box(low=0, high=1,  # todo: normalize it
                                                   shape=(1,), dtype=np.float32)
        elif self.actions == 2:  # place both boxes
            if not self.normalize:
                self.action_space = gym.spaces.Box(low=np.array([-limit1, limit2]), high=np.array([-limit2, limit1]),
                                                   shape=(2,), dtype=np.float32)
            else:
                self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]),
                                                   shape=(2,), dtype=np.float32)
        elif self.actions == 3:  # place three boxes
            if not self.normalize:
                self.action_space = gym.spaces.Box(low=np.array([-limit1, -limit1, limit2]),
                                                   high=np.array([-limit2, -limit2, limit1]),
                                                   shape=(3,), dtype=np.float32)
            else:
                self.action_space = gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]),
                                                   shape=(3,), dtype=np.float32)

        # observation space
        observation_dict = {
            "pos1": Box(low=-20., high=20., shape=(1,), dtype=float),
            "pos2": Box(low=-20., high=20., shape=(1,), dtype=float),
            "angle": Box(low=-0.390258252620697, high=0.390258252620697, shape=(1,), dtype=float),
            # angular velocity of the bar, negative: moves to the right, positive: moves to the left
            "vel": Box(low=-2., high=2., shape=(1,), dtype=float),
            "density1": Box(low=4., high=6., shape=(1,), dtype=float),
            "density2": Box(low=4., high=6., shape=(1,), dtype=float),
            "boxsize1": Box(low=0.8, high=1.2, shape=(1,), dtype=float),
            "boxsize2": Box(low=0.8, high=1.2, shape=(1,), dtype=float),
        } if not self.normalize else {
            "pos1": Box(low=-1., high=1., shape=(1,), dtype=float),
            "pos2": Box(low=-1., high=1., shape=(1,), dtype=float),
            "angle": Box(low=-1, high=1., shape=(1,), dtype=float),
            # angular velocity of the bar, negative: moves to the right, positive: moves to the left
            "vel": Box(low=-1., high=1., shape=(1,), dtype=float),
            "density1": Box(low=0., high=1., shape=(1,), dtype=float),
            "density2": Box(low=0., high=1., shape=(1,), dtype=float),
            "boxsize1": Box(low=0., high=1., shape=(1,), dtype=float),
            "boxsize2": Box(low=0., high=1., shape=(1,), dtype=float),
        }
        if self.number_of_boxes == 3:  # need another added observation for the third box
            if self.normalize:
                observation_dict["pos3"] = Box(low=-1., high=1., shape=(1,), dtype=float)
                observation_dict["density3"] = Box(low=0., high=1., shape=(1,), dtype=float)
                observation_dict["boxsize3"] = Box(low=0., high=1., shape=(1,), dtype=float)
            else:
                observation_dict["pos3"] = Box(low=-20., high=20., shape=(1,), dtype=float)
                observation_dict["density3"] = Box(low=4., high=6., shape=(1,), dtype=float)
                observation_dict["boxsize1"] = Box(low=0.6, high=1., shape=(1,), dtype=float)
                observation_dict["boxsize2"] = Box(low=0.6, high=1., shape=(1,), dtype=float)
                observation_dict["boxsize3"] = Box(low=0.8, high=1.2, shape=(1,), dtype=float)

        self.observation_space = spaces.Dict(spaces=observation_dict)  # convert to Spaces Dict

        """self.observation_space = spaces.Box(low=np.array([-20, -20, -0.390258252620697, -2., 4., 4.]), high=np.array([20, 20, 0.390258252620697, 2., 6., 6.]),
                                           shape=(6,), dtype=np.float32)"""

        # setting up the objects on the screen
        # The ground
        self.ground = self.world.CreateStaticBody(
            shapes=[Box2D.b2.edgeShape(vertices=[(-40, 0), (40, 0)])]
        )

        self.boxes = []
        if self.number_of_boxes == 2:
            self.boxes = self.resetTwoBoxes()
        elif self.number_of_boxes == 3:
            self.boxes = self.resetThreeBoxes()

        topCoordinate = Vec2(0, HEIGHT)
        self.triangle = self.world.CreateStaticBody(
            position=(0, 0),
            fixtures=fixtureDef(shape=polygonShape(vertices=[(-1, 0), (1, 0), topCoordinate]), density=100)
        )

        self.bar = self.world.CreateDynamicBody(
            position=topCoordinate,
            fixtures=fixtureDef(shape=polygonShape(box=(BARLENGTH, 0.3)), density=1),
        )

        self.joint = self.world.CreateRevoluteJoint(bodyA=self.bar, bodyB=self.triangle, anchor=topCoordinate)

        # state calculation
        if self.number_of_boxes == 2:
            # calculate positions = distance to the center of the bar
            pos1 = self.boxA.position[0] / math.cos(self.bar.angle)
            pos2 = self.boxB.position[0] / math.cos(self.bar.angle)
            self.state = np.array([pos1, pos2,
                                   self.bar.angle, self.bar.angularVelocity,
                                   self.randomDensityA if self.random_densities else DENSITY,
                                   self.randomDensityB if self.random_densities else DENSITY,
                                   self.boxsizeA if self.random_boxsizes else BOXSIZE,
                                   self.boxsizeB if self.random_boxsizes else BOXSIZE, ], dtype=np.float32)
        elif self.number_of_boxes == 3:
            pos1 = self.boxA.position[0] / math.cos(self.bar.angle)
            pos2 = self.boxB.position[0] / math.cos(self.bar.angle)
            pos3 = self.boxC.position[0] / math.cos(self.bar.angle)
            self.state = np.array([pos1, pos2, pos3,
                                   self.bar.angle, self.bar.angularVelocity,
                                   self.randomDensityA if self.random_densities else DENSITY,
                                   self.randomDensityB if self.random_densities else DENSITY,
                                   self.randomDensityC if self.random_densities else DENSITY,
                                   self.boxsizeA if self.random_boxsizes else 0.8 * BOXSIZE,
                                   self.boxsizeB if self.random_boxsizes else 0.8 * BOXSIZE,
                                   self.boxsizeC if self.random_boxsizes else BOXSIZE], dtype=np.float32)

        self.normalized_state = None

        self.maxAngle = 0.390258252620697  #self.getMaxAngle() # todo: fix getMaxAngle function
        return

    def ConvertScreenToWorld(self, x, y):
        """
        Returns a b2Vec2 indicating the world coordinates of screen (x,y)
        """
        return Vec2((x + self.viewOffset.x) / self.viewZoom,
                    ((self.screenSize.y - y + self.viewOffset.y) / self.viewZoom))

    def getMaxAngle(self):
        """
        Only called at start to calculate the biggest possible angle.
        Use this to have a good termination condition for the learning process.
        """
        self.maxAngle = math.pi / 2
        if self.number_of_boxes == 3:
            self.deleteBox(self.boxC)
        self.placeBox(self.boxA, - BARLENGTH + 2)
        self.placeBox(self.boxB, BARLENGTH + 3)
        self.internal_step(None)
        while self.bar.angularVelocity != 0:
            self.internal_step(None)
        angle = self.bar.angle
        self.reset()
        return abs(angle)

    def createBox(self, pos_x, pos_y=None, density=DENSITY, boxsize=BOXSIZE):
        """Create a new box on the screen
        Input values: position as x and y coordinate, density and size of the box"""
        try:  # todo: fix
            pos_x = float(pos_x[0])
            pos_y = float(pos_y[0])
        except:
            pass
        if not pos_y:
            pos_y = self.y
        newBox = self.world.CreateDynamicBody(
            position=(pos_x, pos_y),
            fixtures=fixtureDef(shape=polygonShape(box=(boxsize, boxsize)),
                                density=density, friction=1.),
            userData=boxsize,  # save this because you somehow cannot access fixture data later
        )
        self.boxes.append(newBox)
        return newBox

    def deleteBox(self, box):  # todo: fix, maybe ID for every b2Body object
        """Delete a box from the world"""
        if box not in self.boxes:
            print("Box not found")
            return
        pos = self.boxes.index(box)
        self.boxes.pop(pos)
        self.world.DestroyBody(box)
        return

    def deleteAllBoxes(self):
        """Deletes every single box in the world"""
        for box in self.boxes:
            try:
                self.world.DestroyBody(box)
            except Exception as e:
                print(e)
        self.boxes = []
        return

    def resetTwoBoxes(self):
        """Generate the 2 boxes and place them randomly if wished"""
        if self.actions == 1:
            startingPositionA = self.np_random.uniform(- BARLENGTH + 2 * BOXSIZE, - 2 * BOXSIZE)
        else:
            startingPositionA = - BARLENGTH - 3
        startingPositionB = BARLENGTH + 3

        if self.random_densities:
            self.randomDensityA = 4. + 2 * self.np_random.random()  # between 4 and 6
            self.randomDensityB = 4. + 2 * self.np_random.random()

        if self.random_boxsizes:
            self.boxsizeA = self.np_random.uniform(0.8, 1.2)
            self.boxsizeB = self.np_random.uniform(0.8, 1.2)

        self.boxA = self.createBox(pos_x=startingPositionA, pos_y=BOXSIZE,
                                   density=self.randomDensityA if self.random_densities else DENSITY,
                                   boxsize=self.boxsizeA if self.random_boxsizes else BOXSIZE)
        self.boxB = self.createBox(pos_x=startingPositionB, pos_y=BOXSIZE,
                                   density=self.randomDensityB if self.random_densities else DENSITY,
                                   boxsize=self.boxsizeB if self.random_boxsizes else BOXSIZE)

        self.boxes = [self.boxA, self.boxB]
        return self.boxes

    def resetThreeBoxes(self):
        """Generate the 3 boxes and place them randomly if wished"""
        if self.actions == 1:
            startingPositionA = self.np_random.uniform(- BARLENGTH + 2 * BOXSIZE, - 2 * BOXSIZE)
            startingPositionB = self.np_random.uniform(- BARLENGTH + 2 * BOXSIZE, - 2 * BOXSIZE)
            while abs(startingPositionA - startingPositionB) < 0.8 * BOXSIZE:  # try not to place both boxes on one another
                startingPositionB = self.np_random.uniform(- BARLENGTH + 2 * BOXSIZE, - 2 * BOXSIZE)
                print(startingPositionB, startingPositionA-startingPositionB)
        else:
            startingPositionA = - BARLENGTH - 6
            startingPositionB = - BARLENGTH - 3
        startingPositionC = BARLENGTH + 3

        if self.random_densities:
            self.randomDensityA = 4. + 2 * self.np_random.random()  # between 4 and 6
            self.randomDensityB = 4. + 2 * self.np_random.random()
            self.randomDensityC = 4. + 2 * self.np_random.random()

        if self.random_boxsizes:
            self.boxsizeA = self.np_random.uniform(0.6, 1.0)
            self.boxsizeB = self.np_random.uniform(0.6, 1.0)
            self.boxsizeC = self.np_random.uniform(0.8, 1.2)

        self.boxA = self.createBox(pos_x=startingPositionA, pos_y=0.8 * BOXSIZE,
                                   density=self.randomDensityA if self.random_densities else DENSITY,
                                   boxsize=self.boxsizeA if self.random_boxsizes else BOXSIZE)
        self.boxB = self.createBox(pos_x=startingPositionB, pos_y=0.8 * BOXSIZE,
                                   density=self.randomDensityB if self.random_densities else DENSITY,
                                   boxsize=self.boxsizeB if self.random_boxsizes else BOXSIZE)
        self.boxC = self.createBox(pos_x=startingPositionC, pos_y=BOXSIZE,
                                   density=self.randomDensityC if self.random_densities else DENSITY,
                                   boxsize=self.boxsizeC if self.random_boxsizes else BOXSIZE)

        self.boxes = [self.boxA, self.boxB, self.boxC]
        return self.boxes

    def moveBox(self, box, deltaX, deltaY):
        """Move a box in the world along a given vector (deltaX,deltaY)"""
        x = box.position[0] + deltaX
        y = box.position[1] + deltaY
        return self.moveBoxTo(box, x, y)

    def moveBoxTo(self, box, x, y):
        """Place a box at a specific position on the field"""
        self.deleteBox(box)
        boxsize = box.userData  # self.fixedBoxSize
        density = box.mass / (4 * boxsize)  # DENSITY
        movedBox = self.createBox(x, y, density, boxsize)
        return movedBox

    def placeBox(self, box, pos):
        "Place a box on the scale"
        x = math.cos(self.bar.angle) * pos
        y = 6 + math.tan(self.bar.angle) * pos + BOXSIZE
        placedBox = self.moveBoxTo(box, x, y)
        placedBox.angle = self.bar.angle
        return placedBox

    def moveAlongBar(self, box, delta_pos):
        """Take a box and move it along the bar with a """
        # recalculate the position on the bar
        pos = box.position[0] / math.cos(self.bar.angle)
        pos += delta_pos
        return self.placeBox(box, pos)

    def resetState(self):
        """Resets and returns the current values of the state"""
        pos1 = self.boxA.position[0] / math.cos(self.bar.angle)
        pos2 = self.boxB.position[0] / math.cos(self.bar.angle)
        if self.number_of_boxes == 3:
            pos3 = self.boxC.position[0] / math.cos(self.bar.angle)
            self.state = np.array([pos1, pos2, pos3,
                                   self.bar.angle, self.bar.angularVelocity,
                                   self.randomDensityA if self.random_densities else DENSITY,
                                   self.randomDensityB if self.random_densities else DENSITY,
                                   self.randomDensityC if self.random_densities else DENSITY,
                                   self.boxsizeA if self.random_boxsizes else 0.8 * DENSITY,
                                   self.boxsizeB if self.random_boxsizes else 0.8 * DENSITY,
                                   self.boxsizeC if self.random_boxsizes else DENSITY,
                                   ], dtype=np.float32)
        else:
            self.state = np.array([pos1, pos2,
                                   self.bar.angle, self.bar.angularVelocity,
                                   self.randomDensityA if self.random_densities else DENSITY,
                                   self.randomDensityB if self.random_densities else DENSITY,
                                   self.boxsizeA if self.random_boxsizes else DENSITY,
                                   self.boxsizeB if self.random_boxsizes else DENSITY], dtype=np.float32)
        if self.normalize:
            self.normalized_state = self.rescaleState()
        # print(self.state, self.normalized_state)
        return self.state

    def rescaleState(self, state=None):
        """Returns normalized version of the state"""
        """pos1 = self.state[0] / 20.
        pos2 = self.state[1] / 20.
        angle = self.state[2] / self.maxAngle
        angularVelocity = self.state[3] / 2.
        density1 = self.state[4] / 6.
        density2 = self.state[5] / 6."""
        if state is None:
            state = self.state
        if self.number_of_boxes == 2:
            pos1 = rescale_movement([-20., 20.], state[0], [-1, 1])
            pos2 = rescale_movement([-20., 20.], state[1], [-1, 1])
            angle = rescale_movement([-self.maxAngle, self.maxAngle], state[2], [-1, 1])
            angularVelocity = rescale_movement([-2., 2.], state[3], [-1, 1])
            density1 = rescale_movement([0., 6.], state[4], [0, 1])  # other possibility: [4.,6.] -> [-1,1]
            density2 = rescale_movement([0., 6.], state[5], [0, 1])
            boxsize1 = rescale_movement([0.8, 1.2], state[6], [0, 1])
            boxsize2 = rescale_movement([0.8, 1.2], state[7], [0, 1])
            normalized_state = np.array([pos1, pos2,
                                         angle, angularVelocity,
                                         density1, density2,
                                         boxsize1, boxsize2],
                                        dtype=np.float32)
        elif self.number_of_boxes == 3:
            pos1 = rescale_movement([-20., 20.], state[0], [-1, 1])
            pos2 = rescale_movement([-20., 20.], state[1], [-1, 1])
            pos3 = rescale_movement([-20., 20.], state[2], [-1, 1])
            angle = rescale_movement([-self.maxAngle, self.maxAngle], state[3], [-1, 1])
            angularVelocity = rescale_movement([-2., 2.], state[4], [-1, 1])
            density1 = rescale_movement([0., 6.], state[5], [0, 1])  # other possibility: [4.,6.] -> [-1,1]
            density2 = rescale_movement([0., 6.], state[6], [0, 1])
            density3 = rescale_movement([0., 6.], state[7], [0, 1])
            boxsize1 = rescale_movement([0.6, 1.], state[6], [0, 1])
            boxsize2 = rescale_movement([0.6, 1.], state[7], [0, 1])
            boxsize3 = rescale_movement([0.8, 1.2], state[8], [0, 1])
            normalized_state = np.array([pos1, pos2, pos3,
                                         angle, angularVelocity,
                                         density1, density2, density3,
                                         boxsize1, boxsize2, boxsize3],
                                        dtype=np.float32)
        return normalized_state

    def performAction2Boxes(self, action):
        """Place both boxes on the desired positions"""
        # extract information from action
        if self.actions == 1:
            """try:
                box2_pos = action[0]
            except IndexError or TypeError:
                box2_pos = action"""
            box2_pos = action
            if self.normalize:
                box2_pos = rescale_movement([0, 1], box2_pos, [2 * BOXSIZE, BARLENGTH - 2 * BOXSIZE])
        elif self.actions == 2:
            box1_pos = action[0]
            box2_pos = action[1]
            if self.normalize:
                box1_pos = rescale_movement([0, 1], box1_pos, [-BARLENGTH + 2 * BOXSIZE, - 2 * BOXSIZE])
                box2_pos = rescale_movement([0, 1], box2_pos, [2 * BOXSIZE, BARLENGTH - 2 * BOXSIZE])
        # perform action
        if self.actions > 1:
            self.boxA = self.placeBox(self.boxA, box1_pos)
        elif self.actions == 1:
            self.boxA = self.placeBox(self.boxA, self.boxA.position[0])  # now place the box on the scale
        self.boxB = self.placeBox(self.boxB, box2_pos)
        return

    def performAction3Boxes(self, action):
        """Place all 3 boxes on the desired positions"""
        # extract information from action
        print(action)
        if self.actions == 1:
            box3_pos = action
            if self.normalize:
                box3_pos = rescale_movement([0, 1], box3_pos, [2 * BOXSIZE, BARLENGTH - 2 * BOXSIZE])
        elif self.actions == 2:
            box1_pos = action[0]
            box2_pos = action[1]
            box3_pos = action[2]
            if self.normalize:
                box1_pos = rescale_movement([0, 1], box1_pos, [-BARLENGTH + 2 * BOXSIZE, - 2 * BOXSIZE])
                box2_pos = rescale_movement([0, 1], box2_pos, [-BARLENGTH + 2 * BOXSIZE, - 2 * BOXSIZE])
                box3_pos = rescale_movement([0, 1], box3_pos, [2 * BOXSIZE, BARLENGTH - 2 * BOXSIZE])
        # perform action
        if self.actions > 1:
            self.boxA = self.placeBox(self.boxA, box1_pos)
            self.boxB = self.placeBox(self.boxB, box2_pos)
        elif self.actions == 1:
            self.boxA = self.placeBox(self.boxA, self.boxA.position[0])  # now place the box on the scale
            self.boxB = self.placeBox(self.boxB, self.boxB.position[0])
        self.boxC = self.placeBox(self.boxC, box3_pos)

    def step(self, action):
        """Actual step function called by the agent"""
        timesteps = 120
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
        """Simulates the program with the given action and returns the observations"""

        def boxesOnScale():
            """Utility function to check if both boxes are still on the scale"""
            val = len(self.boxA.contacts) >= 1 and len(self.boxB.contacts) >= 1 and len(self.bar.contacts) == 2
            if self.number_of_boxes == 3:
                val = len(self.boxA.contacts) >= 1 and len(self.boxB.contacts) >= 1 and len(self.boxC.contacts) >= 1 and len(self.bar.contacts) == 3
            return val

        def getReward():
            """Calculates the reward and adds it to the self.reward value"""
            # Calculate reward (Scale in balance?)
            if boxesOnScale():
                # both boxes on one side: negative reward
                if self.number_of_boxes == 2 and (not (self.boxA.position[0] < 0 and self.boxB.position[0] > 0)) or self.number_of_boxes == 3 and (not (self.boxA.position[0] < 0 and self.boxB.position[0] < 0 and self.boxC.position[0] > 0)):
                    reward = -1  # (self.maxAngle - abs(self.bar.angle)) / self.maxAngle
                    self.timesteps -= 2
                # box on balance
                elif abs(self.bar.angle) < FAULTTOLERANCE and boxesOnScale():
                    reward = 1
                else:
                    reward = (self.maxAngle - abs(self.bar.angle)) / self.maxAngle
                self.reward += reward
            else:  # one or both boxes not on the scale
                reward = - 1
            return reward

        # Don't do anything if the setting's Hz are <= 0
        hz = 60.
        velocityIterations = 8
        positionIterations = 3
        velocityIterations *= 2
        positionIterations *= 2

        if hz > 0.0:
            timeStep = 1.0 / hz
        else:
            timeStep = 0.0

        self.counter += 1
        self.timesteps += 1

        # check if test failed --> return reward
        if (abs(self.bar.angle) > ANGLE_TRESHOLD * self.maxAngle
                or self.timesteps > MAXITERATIONS):
            state = self.resetState().copy()
            # reward = self.reward / self.timesteps
            # reward = self.timesteps
            reward = getReward()
            self.render()
            self.reset()
            if self.normalize:
                return self.rescaleState(state), reward, True, {}
            return state, reward, True, {}

        # check if no movement anymore
        if self.state[3 + int(self.number_of_boxes == 3)] == 0.0 and boxesOnScale():
            # check if time's up
            if self.counter > WAITINGITERATIONS:
                # self.render()
                state = self.resetState()
                if self.number_of_boxes == 2:
                    print(f"Match: {self.boxA.position[0] / math.cos(self.bar.angle)}\t{self.boxB.position[0] / math.cos(self.bar.angle)}\t{self.bar.angle}\t{20 * math.cos(self.bar.angle)}")
                elif self.number_of_boxes == 3:
                    print(
                        f"Match: {self.boxA.position[0] / math.cos(self.bar.angle)}\t{self.boxB.position[0] / math.cos(self.bar.angle)}\t{self.boxC.position[0]}\t{self.bar.angle}\t{20 * math.cos(self.bar.angle)}")
                reward = getReward()
                self.reset()
                if self.normalize:
                    return self.rescaleState(state), 20 * math.cos(self.bar.angle), True, {}
                return state, 20 * math.cos(self.bar.angle), True, {}
                # return state, 2 * MAXITERATIONS * math.cos(self.bar.angle), True, {}
        else:  # no movement --> reset counter
            self.counter = 0

        # catch special case that no action was executed
        if action is None:
            self.world.Step(timeStep, velocityIterations,
                            positionIterations)
            self.world.ClearForces()
            self.render()
            getReward()
            reward = getReward()
            # reward = self.timesteps
            self.state = self.resetState()
            if self.normalize:
                return self.rescaleState(), reward, False, {}
            return self.state, reward, False, {}

        # Place the boxes
        if self.number_of_boxes == 2:
            self.performAction2Boxes(action=action)
        elif self.number_of_boxes == 3:
            self.performAction3Boxes(action=action)

        # Reset the collision points
        self.points = []

        # Tell Box2D to step
        self.world.Step(timeStep, velocityIterations, positionIterations)
        self.world.ClearForces()

        self.description = f"{self.joint.angle * 180 / math.pi}°"

        self.state = self.resetState()

        # Calculate reward (Scale in balance?)
        reward = getReward()
        # reward = self.timesteps   # old version
        # reward == self.reward / self.timesteps

        # no movement and in balance --> done
        # velocities = [self.bar.linearVelocity, self.boxA.linearVelocity, self.boxB.linearVelocity]
        done = False

        # placeholder for info
        info = {}

        self.render()

        if self.normalize:
            return self.rescaleState(), reward, done, info
        return self.state, reward, done, info

    def close(self):
        pygame.quit()
        sys.exit()

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering
        renderer = self.renderer

        self.screen.fill((0, 0, 0))

        # Set the flags based on what the settings show
        if renderer:
            # convertVertices is only applicable when using b2DrawExtended.  It
            # indicates that the C code should transform box2d coords to screen
            # coordinates.
            is_extended = isinstance(renderer, b2DrawExtended)
            renderer.flags = dict(drawShapes=True,
                                  drawJoints=False,  # True
                                  drawAABBs=False,
                                  drawPairs=False,
                                  drawCOMs=False,
                                  convertVertices=is_extended,
                                  )

        self.world.warmStarting = True
        self.world.continuousPhysics = True
        self.world.subStepping = False

        # Reset the collision points
        self.points = []

        if renderer is not None:
            renderer.StartDraw()

        self.world.DrawDebugData()

        if renderer:
            renderer.EndDraw()
            pygame.display.flip()

    def reset(self):
        self.deleteAllBoxes()

        if self.number_of_boxes == 2:
            self.boxes = self.resetTwoBoxes()
        elif self.number_of_boxes == 3:
            self.boxes = self.resetThreeBoxes()

        # rearrange the bar to 0 degree
        self.bar.angle = 0
        self.bar.angularVelocity = 0.

        # Reset the reward and the counters
        self.counter = 0
        self.timesteps = 0
        self.reward = 0

        # return the observation
        self.resetState()
        return self.rescaleState() if self.normalize else self.state  #todo: check for mistakes
        # return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


# More functions can be changed to allow for contact monitoring and such.
# See the other testbed examples for more information.

if __name__ == "__main__":
    main(ScaleExperiment)
