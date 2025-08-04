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


class Scale(Framework, gym.Env):
    """You can use this class as an outline for your tests."""
    name = "ScaleExperiment"  # Name of the class to display

    def __init__(self, rendering=True, random_densities=True, random_boxsizes=False, normalize=False, placed=1,
                 actions=1, sides=2, raw_pixels=False):
        super(Scale, self).__init__(rendering)

        self.np_random = None
        self.seed()

        self.num_envs = 1  # for stable-baseline3

        # Initialize all of the objects
        self.y = 6.0 + BOXSIZE
        BARHEIGHT = 6  # height of the bar joint

        # screen / observation space measurements
        self.height = 480
        self.width = 640

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

        # action space determination
        limit1, limit2 = BARLENGTH - 2 * BOXSIZE, 2 * BOXSIZE
        if not self.normalize:
            self.action_space = gym.spaces.Box(
                low=np.array([-limit1 if not self.sides == 1 else limit2 for _ in range(actions)]),
                high=np.array([limit1 for _ in range(actions)]),
                shape=(self.actions,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(low=np.array([-1 if self.sides == 2 else 0 for _ in range(actions)]),
                                               high=np.array([1 for _ in range(actions)]),
                                               shape=(1,), dtype=np.float32)

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
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, 3), dtype=np.uint8)

        """self.observation_space = spaces.Box(low=np.array([-20, -20, -0.390258252620697, -2., 4., 4.]), high=np.array([20, 20, 0.390258252620697, 2., 6., 6.]),
                                           shape=(6,), dtype=np.float32)"""

        # setting up the objects on the screen
        self.ground = self.world.CreateStaticBody(
            shapes=[Box2D.b2.edgeShape(vertices=[(-40, 0), (40, 0)])]
        )

        self.maxAngle = 0.390258252620697  # self.getMaxAngle() # todo: fix getMaxAngle function

        topCoordinate = Vec2(0, BARHEIGHT)
        self.triangle = self.world.CreateStaticBody(
            position=(0, 0),
            fixtures=fixtureDef(shape=polygonShape(vertices=[(-1, 0), (1, 0), topCoordinate]), density=100)
        )

        self.bar = self.world.CreateDynamicBody(
            position=topCoordinate,
            fixtures=fixtureDef(shape=polygonShape(box=(BARLENGTH, 0.3)), density=1),
        )

        self.joint = self.world.CreateRevoluteJoint(bodyA=self.bar, bodyB=self.triangle, anchor=topCoordinate)

        self.boxes = {}
        self.boxsizes = {}
        # self.resetBoxes()
        self.reset()

        # state calculation
        self.state = None
        self.internal_state = None
        self.state = self.resetState()

        self.normalized_state = None
        return

    def ConvertScreenToWorld(self, x, y):
        """
        Returns a b2Vec2 indicating the world coordinates of screen (x,y)
        """
        return Vec2((x + self.viewOffset.x) / self.viewZoom,
                    ((self.screenSize.y - y + self.viewOffset.y) / self.viewZoom))

    def convertDensityToRGB(self, density, low=4., high=6.):
        """
        Gets a value for the density of one box and returns the corresponding color
        :param density: the given
        :param low: the minimum value for the density
        :param high: the maximum value for the density
        :return: a RGB color
        """
        if not (low <= density <= high):
            raise AssertionError(f"Density {density} not in allowed range [{low},{high}]")

        # first normalize the density
        density = int(rescale_movement([low, high], density, [0., 256**3 - 1]))
        #print(density)

        red = (density >> 16) & 255
        green = (density >> 8) & 255
        blue = density & 255

        return red, green, blue

    def convertRGBToDensity(self, RGB, low=4., high=6.):
        """
        Gets a RGB value of an box and returns the corresponding density of the box
        :param RGB: (red, green, blue) values
        :param low: the minimum value for the density
        :param high: the maximum value for the density
        :return: density value
        """
        red, green, blue = RGB[0], RGB[1], RGB[2]
        density = (1/256**3) * (256**2 + red + 256 * green + blue)

        # rescale the density
        density = rescale_movement([0., 256**3 - 1], density, [low, high])

        return density

    def getMaxAngle(self):
        """
        Only called at start to calculate the biggest possible angle.
        Use this to have a good termination condition for the learning process.
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

    def createBox(self, pos_x, pos_y=None, density=DENSITY, boxsize=BOXSIZE, index=0, static=False):
        """Create a new box on the screen
        Input values: position as x and y coordinate, density and size of the box"""
        try:  # todo: fix
            pos_x = float(pos_x[0])
            pos_y = float(pos_y[0])
        except:
            pass
        if not pos_y:
            pos_y = self.y
        if static:
            newBox = self.world.CreateStaticBody(
                position=(pos_x, pos_y),
                fixtures=fixtureDef(shape=polygonShape(box=(boxsize, boxsize)),
                                    density=density, friction=1.),
                # userData=(255,255,255)#self.convertDensityToRGB(density=density)
                userData=boxsize,  # save this because you somehow cannot access fixture data later
            )
        else:
            newBox = self.world.CreateDynamicBody(
                position=(pos_x, pos_y),
                fixtures=fixtureDef(shape=polygonShape(box=(boxsize, boxsize)),
                                    density=density, friction=1.),
                # userData=(255,255,255)#self.convertDensityToRGB(density=density)
                userData=boxsize,  # save this because you somehow cannot access fixture data later
            )
        if index == 0:
            index = len(self.boxes.values()) + 1
        self.boxes[index] = newBox
        self.boxsizes[index] = boxsize
        return newBox

    def deleteBox(self, box):
        """Delete a box from the world"""
        if box not in self.boxes.values():
            print("Box not found")
            return
        key = list(self.boxes.keys())[list(self.boxes.values()).index(box)]
        del self.boxes[key]
        self.world.DestroyBody(box)
        return

    def deleteAllBoxes(self):
        """Deletes every single box in the world"""
        for box in self.boxes.values():
            try:
                self.world.DestroyBody(box)
            except Exception as e:
                print(e)
        self.boxsizes = {}
        self.boxes = {}
        return

    def resetBoxes(self):
        """Generate the boxes and place all of the ones that are supposed to be placed randomly"""

        def overlapping(boxes=None):
            """Help function to determine whether two boxes would overlap when trying to place them to their
            starting positions given their sizes"""
            if boxes is None:
                boxes = []
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
        self.boxes = {}
        self.boxsizes = {}
        self.densities = {}
        self.boxsizes = {}
        self.positions = {}
        i = 0
        for (i, (pos, size)) in enumerate(boxes, start=1):
            density = 4. + 2 * self.np_random.random()  # between 4 and 6
            box = self.createBox(pos_x=pos, pos_y=size,
                                 density=density if self.random_densities else DENSITY,
                                 boxsize=size if self.random_boxsizes else BOXSIZE,
                                 index=i, static=False)
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
                                 index=index, static=False)     # todo: fix problem with static objects
            self.boxes[index] = box
            self.densities[index] = density if self.random_densities else DENSITY
            self.boxsizes[index] = boxsize if self.random_boxsizes else BOXSIZE
            self.positions[index] = position
        return self.boxes

    def moveBox(self, box, deltaX, deltaY, index=0):
        """Move a box in the world along a given vector (deltaX,deltaY)"""
        x = box.position[0] + deltaX
        y = box.position[1] + deltaY
        return self.moveBoxTo(box, x, y, index)

    def moveBoxTo(self, box, x, y, index=0):
        """Place a box at a specific position on the field"""
        self.deleteBox(box)
        # boxsize = box.userData  # self.fixedBoxSize
        boxsize = self.boxsizes[index]
        density = self.densities[index]  # DENSITY
        movedBox = self.createBox(x, y, density, boxsize, index=index, static=False)
        return movedBox

    def placeBox(self, box, pos, index=0):
        "Place a box on the scale"
        x = math.cos(self.bar.angle) * pos
        y = 6 + math.tan(self.bar.angle) * pos + BOXSIZE
        placedBox = self.moveBoxTo(box, x, y, index=index)
        placedBox.angle = self.bar.angle
        return placedBox

    def moveAlongBar(self, box, delta_pos, index=0):
        """Take a box and move it along the bar with a """
        # recalculate the position on the bar
        pos = box.position[0] / math.cos(self.bar.angle)
        pos += delta_pos
        return self.placeBox(box, pos, index=index)

    def resetState(self):
        """Resets and returns the current values of the state"""
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
            """self.screen = pygame.display.set_mode((self.width, self.height))
            # overwrite the state with the pixel 3d array
            self.state = pygame.surfarray.array3d(pygame.display.get_surface())
            self.screen = pygame.display.set_mode((1, 1))"""

            # self.state = self.render("state_pixels")
            self.state = self._create_image_array(self.screen, (self.width, self.height))

        if self.normalize:
            self.normalized_state = self.rescaleState()
        return self.state

    def rescaleState(self, state=None):
        """Returns normalized version of the state"""
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
        if self.normalize:
            if self.sides == 1:
                actions = rescale_movement([0., 1.], actions, [2 * BOXSIZE, BARLENGTH - 2 * BOXSIZE])
            else:
                actions = rescale_movement([-1., 1.], actions, [-BARLENGTH + 2 * BOXSIZE, BARLENGTH - 2 * BOXSIZE])

        i = 0
        for i in range(1, self.placed + 1):
            self.boxes[i] = self.placeBox(self.boxes[i], self.boxes[i].position[0], index=i)
        try:
            for j, action in enumerate(actions):
                self.boxes[i + j + 1] = self.placeBox(self.boxes[i + j + 1], action, index=i + j + 1)
        except:
            self.boxes[i + 1] = self.placeBox(self.boxes[i + 1], actions, index=i + 1)
        return

    def step(self, action):
        """Actual step function called by the agent"""
        timesteps = 120
        self.action = action
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
            boxes = np.array(list(self.boxes.values()))
            for box in list(self.boxes.values()):
                if len(box.contacts) < 1:
                    return False
            val = len(self.bar.contacts) == self.actions + self.placed
            return True  # val

        def getReward():
            """Calculates the reward and adds it to the self.reward value"""
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
        hz = 60.
        velocityIterations = 8
        positionIterations = 3
        velocityIterations *= 1
        positionIterations *= 1

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
            reward = getReward()
            self.render()
            # self.render(mode="state_pixels" if self.raw_pixels else "human")
            self.reset()
            if self.normalize:
                return self.rescaleState(state), reward, True, {}
            return state, reward, True, {}

        # check if no movement anymore
        if self.internal_state[1 + self.actions + self.placed] == 0.0 and boxesOnScale():
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
        self.performActions(actions=action)

        # Reset the collision points
        self.points = []

        # Tell Box2D to step
        self.world.Step(timeStep, velocityIterations, positionIterations)
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
        pygame.quit()
        sys.exit()

    def render(self, mode="human"):
        assert mode in ["human", "rgb_array", "state_pixels"], f"Wrong render mode passed, {mode} is invalid."
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

        if mode == "rgb_array":
            return self._create_image_array(self.screen, (self.width, self.height))

        elif mode == "state_pixels":
            return self._create_image_array(self.screen, (self.width, self.height))

        elif mode == "human":
            if renderer is not None:
                renderer.StartDraw()

            self.world.DrawDebugData()

            if renderer:
                renderer.EndDraw()
                pygame.display.flip()

            return True  # ?

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.array(pygame.surfarray.pixels3d(scaled_screen))

    def reset(self):
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
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


# More functions can be changed to allow for contact monitoring and such.
# See the other testbed examples for more information.

if __name__ == "__main__":
    main(Scale)
