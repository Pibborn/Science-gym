import abc
import sys
from abc import ABC
from time import sleep
from typing import TypeVar, Generic, Tuple, Union, Optional, SupportsFloat

import gym
from gym import spaces
from gym.spaces import Box
import pygame
from Box2D import b2Vec2
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, kinematicBody, edgeShape, fixtureDef)
import numpy as np
from gym.core import ActType, ObsType
from gym.utils import seeding

PPM = 16  # 12  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480  # 480, 360

colors = {
    staticBody: (255, 255, 255, 255),
    dynamicBody: (127, 127, 127, 255),
    kinematicBody: (0, 127, 127, 255),
}
groundColor = (255, 255, 255)


def rescale_movement(original_interval, value, to_interval=(-1, +1)):
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


class EnvironmentInterface(gym.Env, ABC):
    def __init__(self, seed=None, normalize=False, rendering=False, raw_pixels=False, walls=0, gravity=-9.80665):
        self.seed(seed)

        self.num_envs = 1  # for stable-baseline3

        # pygame stuff
        self.screen_height = SCREEN_HEIGHT
        self.screen_width = SCREEN_WIDTH

        # Pygame setup
        if rendering:
            pygame.init()

            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), 0, 32)
            pygame.display.set_caption('pygame setup')
        else:
            self.screen = pygame.display.set_mode((1, 1))
        self.clock = pygame.time.Clock()

        self.timesteps = 0
        self.max_timesteps = 300

        # set up the box2d world
        self.world = world(gravity=(0, gravity), doSleep=True)
        self.world_height = 30  # todo: set it up right
        self.world_width = 40

        # setting up the objects on the screen
        self.ground = self.world.CreateStaticBody(
            position=(self.world_width/2, 0),
            shapes=polygonShape(box=(self.world_width, 1)),
            userData=groundColor,
        )

        if walls > 0:
            self.right_wall = self.world.CreateStaticBody(
                position=(self.world_width, 0),
                shapes=polygonShape(box=(1, self.world_height)),
                userData=groundColor,
            )
        if walls > 1:
            self.left_wall = self.world.CreateStaticBody(
                position=(0, 0),
                shapes=polygonShape(box=(1, self.world_height)),
                userData=groundColor,
            )


        self.normalize = normalize
        self.rendering = rendering  # should the simulation be rendered or not
        self.raw_pixels = raw_pixels  # use pixels for state or not

        # if we use Dicts as observation space and want to clarify in which order we want to use the values in the state
        self.order = None

        # gym action and observation spaces
        # self.action_space = gym.spaces.Box(low=np.array([-np.pi, 0]), high=np.array([np.pi, 100]))
        # self.observation_space = gym.spaces.Box(low=np.array([-np.pi, 0]), high=np.array([np.pi, 100]))
        self.action_space = None
        self.observation_space = Box(low=0, high=1. if self.normalize else 255,
                                     shape=(self.world_width, self.world_height,
                                            3),
                                     dtype=np.float32 if self.normalize else np.uint8) if self.raw_pixels else None

        self.state = None
        self.old_state = None
        self.normalized_state = None
        self.reset()

    def step(self, action=None):
        done = False
        # self.state = self.reset()
        for _ in range(self.max_timesteps):
            # self.render()
            self.old_state = self.state
            self.state, reward, done, info = self.internal_step(action)
            action = None
            if done:
                break
        if not done:
            done = True
            self.reset()
        return self.state, reward, done, info

    @abc.abstractmethod
    def success(self):
        """
        Has the experiment succeeded?

        :return: True if success, else False
        :rtype: bool
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getReward(self):
        """

        :return:
        """
        raise NotImplementedError

    def testFailed(self):
        """
        Only overwrite, if there is an opportunity that the experiment might be at a point where it cannot
        succeed anymore.

        :return: True, if the test fails, else False
        :rtype: bool
        """
        return False

    def printSuccess(self):
        """
        return message, if we have a match or a successful outcome

        :return:
        :rtype: str
        """
        return ""

    def internal_step(self, action=None):
        velocityIterations = 8
        positionIterations = 3
        velocityIterations *= 1
        positionIterations *= 1

        """
        if action is not None:
            self.performAction(action)
        self.world.Step(TIME_STEP, velocityIterations,
                        positionIterations)
        self.world.ClearForces()"""

        # check if test failed --> return reward
        if self.testFailed():
            state = self.updateState()
            reward = self.getReward()
            self.render()
            # self.render(mode="state_pixels" if self.raw_pixels else "human")
            self.reset()
            return state if not self.normalize else self.rescaleState(state), reward, True, {}

        # check if success
        if self.success():
            # self.render()
            state = self.updateState()
            if self.printSuccess() != "":
                print(self.printSuccess())
            reward = self.getReward()
            # self.reset()
            return state if not self.normalize else self.rescaleState(state), reward, True, {}

        # catch special case that no action was executed
        if action is None:
            self.world.Step(TIME_STEP, velocityIterations,
                            positionIterations)
            self.world.ClearForces()
            # self.render(mode="state_pixels" if self.raw_pixels else "human")
            self.render()
            reward = self.getReward()
            self.state = self.updateState()
            return self.state if not self.normalize else self.rescaleState(), reward, False, {}

        # do whatever is done in the step function
        self.performAction(action=action)

        # Tell Box2D to step
        self.world.Step(TIME_STEP, velocityIterations, positionIterations)
        self.world.ClearForces()

        self.state = self.updateState()

        # Calculate reward
        reward = self.getReward()
        done = False

        # placeholder for info
        info = {}

        self.render()
        # self.render(mode="state_pixels" if self.raw_pixels else "human")

        return self.state if not self.normalize else self.rescaleState(), reward, done, info

    @abc.abstractmethod
    def performAction(self, action):
        """
        perform the actual action from the step function

        :param action: ActType
        :return:
        """
        raise NotImplementedError

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
            vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
            # vertices = [b2Vec2(x * PPM, self.screen_height - y * PPM) for (x, y) in polygon.vertices]
            # vertices = [b2Vec2(self.screen_width - x * PPM, self.screen_height - y * PPM) for (x, y) in polygon.vertices]
            """vertices2 = [v * PPM for v in polygon.vertices]
            # vertices = [(v[0] + SCREEN_WIDTH / 2, SCREEN_HEIGHT - v[1]) for v in vertices]
            
            print(vertices)
            print(polygon.vertices)"""
            if body.userData is not None:
                pygame.draw.polygon(self.screen, body.userData, vertices)
            else:
                # pygame.draw.polygon(self.screen, self.convertDensityToRGB(density=fixture.density), vertices)
                # here: don't use the red color channel, only use green and blue
                pygame.draw.polygon(self.screen,
                                    colors[body.type],
                                    # self.convertDensityToRGB(density=fixture.density, channels=[False, True, False]),
                                    vertices)

        polygonShape.draw = my_draw_polygon

        def my_draw_circle(circle, body, fixture):
            # position = body.transform * circle.pos * PPM
            # position = (circle.pos[0] * PPM, self.screen_height - circle.pos[1] * PPM)
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
            return self._create_image_array(self.screen, (self.screen_width, self.screen_height))

        elif mode == "state_pixels":  # todo: check if it works
            return self._create_image_array(self.screen, (self.screen_width, self.screen_height))

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

    @abc.abstractmethod
    def updateState(self):
        """
        Reset the state

        :return: updated state
        :rtype: ??
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rescaleState(self, state=None):
        """
        Normalize the state

        :param state: State that should be normalized
        :type state: ??
        :return: nromalized state
        :rtype: ??
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    """def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        pass"""

    def close(self):
        """Close the pygame window and terminate the program."""
        pygame.quit()
        sys.exit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
