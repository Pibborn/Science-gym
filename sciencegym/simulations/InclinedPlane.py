import math
import sys
import random

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

import xml.etree.ElementTree as ET

# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 20.0  # pixels per meter
TARGET_FPS = 60  # to render slowly set target fps to 200 and uncomment clock.tick
demo = False
if demo:
    TARGET_FPS = 30
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

BALL_RADIUS = 1

FAULTTOLERANCE = 0.001  # for the angle of the bar
ANGLE_TRESHOLD = 0.98

WAITINGITERATIONS = 10  # maximum iterations to wait per episode
MAXITERATIONS = 1000


def normal_vector(p1, p2):
    d = (p2[0] - p1[0], p2[1] - p1[1])
    n = (d[1], -d[0])
    norm = np.linalg.norm(n)
    return (abs(n[0] / norm), abs(n[1] / norm))


def degreeToRad(angle):
    """
    :param angle:
    :return:
    """
    return angle / 180 * np.pi


def radToDegree(rad):
    """
    :param rad:
    :return:
    """
    return rad / np.pi * 180


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


class InclinePlaneCoordinates():
    def __init__(self, bottom_left, bottom_right, top_left):
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.top_left = top_left

    def __init__(self, angle: float, l: float = 20):

        """
        :param l: float in cm constant (10m = 1000cm)
        :param angle: in degree between 0 and 90 degree
        A = left upper corner, B= left lower  corner, C = right lower corner, D = middle of inclined plane surface = here is ball initially placed
        :return: [A=(x,y), B=(x2,y2), C=(x3,y3), D= (x4, y4)]
        """
        self.angle = angle # angle in deg
        angle = degreeToRad(angle)
        cosAngle = np.cos(angle)
        sinAngle = np.sin(angle)
        A = (-l * cosAngle / 2, l * sinAngle)
        B = (-l * cosAngle / 2, 0)
        # C = Vec2(l * cosAngle / 2, 0)
        C = (l * cosAngle / 2, 0)

        # calculate ball position, moved along the normal vecor of plane at length radius to set ball on plane

        n = normal_vector(A, C) * BALL_RADIUS
        D = (n[0], n[1] + (l * sinAngle / 2))
        if self.angle <90:
            self.bottom_left = B
            self.bottom_right = C
            self.top_left = A
            self.length = l
            self.initial_ball_position = D
        else:
            self.bottom_left = C
            self.bottom_right = B
            self.top_left = A
            self.length= l
            self.initial_ball_position = D


    def get_coordinates(self):
        return [self.top_left, self.bottom_left, self.bottom_right, self.initial_ball_position]

    def is_x_coordinate_inside_plane_bounds(self, x):
        if x > self.bottom_left[0] and x < self.bottom_right[0]:
            return True
        return False

    def is_y_coordinate_inside_plane_bounds(self, y):
        if y > 0 and y < self.top_left[1] + BALL_RADIUS:
            return True
        return False

    def is_ball_in_plane_bounds(self, coordinates):
        if (self.is_x_coordinate_inside_plane_bounds(coordinates[0])
                and self.is_y_coordinate_inside_plane_bounds(coordinates[1])):
            return True
        return False

    def abs_ball_distance_form_initial_positione(self, ball_coords):
        return np.linalg.norm(
            (ball_coords[0] - self.initial_ball_position[0], ball_coords[1] - self.initial_ball_position[1]))

    def rel_ball_distance_form_initial_positione(self, ball_coords):
        return np.linalg.norm(
            (ball_coords[0] - self.initial_ball_position[0], ball_coords[1] - self.initial_ball_position[1])) / (
                    self.length / 2)


class InclinedPlane(gym.Env):
    name = "InclinedPlane-v1"  # Name of the class to display

    def __init__(self, args, rendering=True, random_densities=False, random_boxsizes=False, normalize=False, placed=1,
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

        # hilfsvariablen --------------

        # ------------------------------
        self.args
        self.np_random = None
        self.seed()
        self.num_envs = 1  # for stable-baseline3
        self.name = "InclinedPlane-v1"

        # screen / observation space measurements
        self.height = SCREEN_HEIGHT  # 480
        self.width = SCREEN_WIDTH  # 640

        # Pygame setup
        if rendering:
            pygame.init()

            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('Inclined Plane')
        else:
            self.screen = pygame.display.set_mode((1, 1))
        self.clock = pygame.time.Clock()

        self.counter = 0
        self.timesteps = 0
        self.reward = 0

        self.rendering = rendering  # should the simulation be rendered or not
        self.color_counter = 0
        self.normalize = normalize

        self.raw_pixels = raw_pixels
        self.order = None

        # determine observation and action space here
        # gravities
        gravity_earth = -9.80665
        gravity_moon = -1.62
        gravity_mars = -3.71
        gravity_venus = -8.87
        self.gs_list = [-5, -6, -7, -8, -9] #, gravity_venus, -5, -5.5, -6.5, -7.5, -8.5, -9.5 ,-6, -7, -9, -10] # , gravity_mars, gravity_moon]
        self.min_gravity = min(self.gs_list)
        self.max_gravity= max(self.gs_list)

        # weights
        self.min_mass = 5
        self.max_mass = 2000

        # angles
        self.min_angle = 2
        self.max_angle = 89

        # forces
        min_force = 0
        max_force = abs(self.max_mass * self.max_gravity)

        # Box2d world setup
        # Create the world
        self.world = world(gravity=(0, -9.80665), doSleep=True)


        if not self.normalize:
            self.action_space = Box(
                low=min_force,
                high=max_force,
                shape=(1,), dtype=np.float32)

        else:
            self.action_space = Box(
                low=min_force,
                high=max_force,
                shape=(1,), dtype=np.float32)

        # observation space

        if not raw_pixels:
            observation_dict = {
                "mass": Box(low=self.min_mass, high=self.max_mass, shape=(1,), dtype=np.float32),
                "gravity": Box(low=self.min_gravity, high=self.max_gravity, shape=(1,), dtype=np.float32),
                "angle": Box(low=self.min_angle, high=self.max_angle, shape=(1,), dtype=np.float32),
                "force": Box(low=min_force, high=max_force, shape=(1,), dtype=np.float32)

            } if not self.normalize else {
                "mass": Box(low=self.min_mass, high=self.max_mass, shape=(1,), dtype=int),
                "gravity": Box(low=self.min_gravity, high=self.max_gravity, shape=(1,), dtype=np.float32),
                "angle": Box(low=self.min_angle, high=self.max_angle, shape=(1,), dtype=int),
                "force": Box(low=min_force, high=max_force, shape=(1,), dtype=np.float32)
            }

            self.observation_space = spaces.Dict(spaces=observation_dict)  # convert to Spaces Dict

        else:
            dummy_obs = self.render("rgb_array")
            self.observation_space = spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

        # set inital force to zero for state
        self.force = 0

        # reset every dict/array and all the boxes on the screen
        self.reset()

        # state calculation
        self.internal_state = None
        self.state = self.getState()
        self.normalized_state = None

        return None

    def createNewExperiment(self):

        def random_rgb_value():
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # world is created in innit so is the ground
        # Reset gravity of world
        self.gravity = random.choice(self.gs_list)
        self.world.gravity = (0, self.gravity)

        # create inclined plane
        # no angle equal 90 degree permitted
        self.angle = np.random.randint(self.min_angle, self.max_angle + 1)
        while 89.1<= self.angle <= 90.9:
            self.angle = np.random.randint(self.min_angle, self.max_angle + 1)
        try:
            self.world.DestroyBody(self.inclinedPlane)
        except Exception as e:
            print(f"destroy inclined plane{e}")
        # destroy old ball and create new ball
        try:
            self.world.DestroyBody(self.ball)
        except Exception as e:
            print(f"destroy inclined plane{e}")

        # coordinates = coordinatesInclinedPlane(self.angle)
        # self.plane_cords = InclinePlaneCoordinates(coordinates[1], coordinates[2], coordinates[0])

        self.plane_cords = InclinePlaneCoordinates(self.angle)
        coordinates = self.plane_cords.get_coordinates()

        self.ball = self.world.CreateDynamicBody(
            position=coordinates[3],
            fixtures=fixtureDef(shape=circleShape(radius=BALL_RADIUS),
                                density=(random.uniform(self.min_mass, self.max_mass) / math.pi))
        )

        color = random_rgb_value()

        # Einkommentieren um Töne abzuspielen
        # play_sound_from_rgb(color)

        self.inclinedPlane = self.world.CreateStaticBody(
            position=(0, 0),
            fixtures=fixtureDef(shape=polygonShape(vertices=coordinates[0:3]),
                                density=100),
            userData=color)

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

    def getState(self):
        """
        Resets and returns the current values of the state

        get current

        :return: the new state
        :rtype: np.ndarray
        """
        self.mass = self.ball.mass
        self.state = np.array([self.ball.mass, self.gravity, degreeToRad(self.angle), self.force], dtype=np.float32)
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
            normalized_state = None
        return normalized_state
    def step(self, action):
        """
        here you can exchange the step function very simple from simulated to synthetic step functions
        options:
        1) simulated steps
        - simulated_step (here you can exchange the reward function in the internal step function)
        2) synthetic steps
        - synthetic_step_percentual_difference_between_forces
        - synthetic_step_percentual_change_position_or_velocity
        - step_without_physics_constant
        - step_without_physics_angle_dependant
        :param action:
        :return:
        """
        return self.synthetic_step_percentual_change_position_or_velocity(action)

    def simulated_step(self, action):
        """
        Actual step function called by the agent when box2d simulation is used

        :param action: the action(s) the agent chooses as an array of each new positions for each box that should be moved
        :type action: list[float]
        :return: new state after step, the reward, done, info
        :rtype: tuple[np.ndarray, float, bool, dict]
        """
        self.force = action[0]
        # print(f'action: in step function \n {action[0]}')
        self.getState()

        timesteps = 60

        done = False
        for _ in range(timesteps):
            self.old_state = self.state
            self.state, reward, done, info = self.internal_step(action)
            # todo: maybe keep action going (we apply force all time)
            # action = None # todo: when action = None apply change to internal-step
            if done:
                break
        # if not done:
        #    done = True
        #   self.reset()
        state = self.state.copy()
        self.reset()
        done = True
        return state, reward, done, info

    def synthetic_step_percentual_difference_between_forces(self, action):
        """
        gets reward dependant on how much percentual difference is between the force applied by the agent
        and the actual force needed
        :param action:
        :return:
        """
        self.getState()
        Fagent = action[0]
        degangle = self.angle
        radangle = degreeToRad(degangle)
        gravity = 9.80665
        mass = self.mass
        Fgoal = mass * gravity * math.sin(radangle)
        relDifference = (abs(Fagent - Fgoal) / abs(Fgoal))
        if  0<= relDifference <= 0.001:
            reward = 2
        elif relDifference <= 0.003:
            reward = 1
        elif relDifference <= 0.01:
            reward = 0
        else:
            reward = -1
        self.old_state = self.state
        state = np.array([mass, gravity, radangle, Fagent], dtype=np.float32)
        self.state = state
        return state, reward, True, {}

    def synthetic_step_percentual_change_position_or_velocity(self, action):
        """
        gets reward dependant on how much percentual difference is between the observed movement or velocity
        of the ball to the movement or velocity you would expect if the force was predicted correctly
        with a percentual difference of delta_percent
        :param action:
        :return:
        """
        delta_percent = 0.014
        t = 1
        self.getState()
        Fagent = action[0]
        degangle = self.angle
        radangle = degreeToRad(degangle)
        gravity = round(random.uniform(6, 12), 2) # 9.80665
        mass = self.mass

        Fgoal = mass * gravity * math.sin(radangle)
        accgoal = gravity * math.sin(radangle) * delta_percent
        vgoal = accgoal * t
        sgoal = 1/2 * accgoal * t**2

        Fres = abs(Fgoal - Fagent)
        accagent = Fres/mass # a
        vagent = accagent * t
        sagent = 1/2 * accagent * t**2

        isstate = sagent
        goalstate = sgoal
        if 0 <= isstate <= goalstate:
            reward = 2
        elif isstate <= 3*goalstate:
            reward = 1
        elif isstate <= 10*goalstate:
            reward = 0
        else:
            reward = -1
        self.old_state = self.state
        state = np.array([mass, gravity, radangle, Fagent], dtype=np.float32)
        self.state = state
        return state, reward, True, {}

    def step_without_physics_constant(self, action):
        """
        like synthetic_step_percentual_change_position_or_velocity to play around with the constants and parameters
        here the gravity is set constant
        :param action:
        :return:
        """
        delta_percent = 0.03 # 0.018
        t = 1
        self.getState()
        Fagent = action[0]
        degangle = self.angle
        radangle = degreeToRad(degangle)
        gravity = 9.80665
        mass = self.mass

        Fgoal = mass * gravity * math.sin(degreeToRad(30))
        accgoal = gravity * math.sin(degreeToRad(30)) * delta_percent
        vgoal = accgoal * t
        sgoal = 1/2 * accgoal * t**2

        Fres = abs(Fgoal - Fagent)
        accagent = Fres/mass # a
        vagent = accagent * t
        sagent = 1/2 * accagent * t**2

        isstate = sagent
        goalstate = sgoal
        if 0 <= isstate <= goalstate:
            reward = 2
        elif isstate <= 3*goalstate:
            reward = 1
        elif isstate <= 10*goalstate:
            reward = 0
        else:
            reward = -1
        self.old_state = self.state
        state = np.array([mass, gravity, radangle, Fagent], dtype=np.float32)
        self.state = state
        return state, reward, True, {}

    def step_without_physics_angle_dependant(self, action):
        """
        try to learn small angles as good as bigger angles
        by decreasing the allowed change in position of the ball the smaller the angle is
        :param action:
        :return:
        """
        delta_percent = 0.018
        t = 1
        self.getState()
        Fagent = action[0]
        degangle = self.angle
        radangle = degreeToRad(degangle)
        gravity = 9.80665
        mass = self.mass

        Fgoal = mass * gravity * math.sin(degreeToRad(2))
        accgoal = gravity * math.sin(degreeToRad(2)) * delta_percent
        vgoal = accgoal * t
        sgoal = 1/2 * accgoal * t**2

        Fres = abs(Fgoal - Fagent)
        accagent = Fres/mass # a
        vagent = accagent * t
        sagent = 1/2 * accagent * t**2

        isstate = sagent
        goalstate = sgoal

        if radangle < 1:
            goalstate *= radangle
        if 0 <= isstate <= goalstate:
            reward = 2
        elif isstate <= 3*goalstate:
            reward = 1
        elif isstate <= 10*goalstate:
            reward = 0
        else:
            reward = -1
        self.old_state = self.state
        state = np.array([mass, gravity, radangle, Fagent], dtype=np.float32)
        self.state = state
        return state, reward, True, {}

    def internal_step(self, action=None):
        """
        Simulates the program with the given action and returns the observations

        :param action: the action(s) the agent chooses as an array of each new positions for each box that should be moved
        :type action: list[float]
        :return: new state after step, the reward, done, info
        :rtype: tuple[np.ndarray, float, bool, dict]
        """
        velocityIterations = 8
        positionIterations = 3

        def getBallVelocity():
            vel = np.linalg.norm(self.ball.linearVelocity)
            # print(f"ball velocity: {vel}")
            return -1 * vel

        def ball_distance_reward():
            if not self.plane_cords.is_ball_in_plane_bounds(self.ball.position):
                return -10
            rel_delta = self.plane_cords.rel_ball_distance_form_initial_positione(self.ball.position)
            return 2 - rel_delta * 10

        def better_ball_distance_reward():
            abs_delta = self.plane_cords.abs_ball_distance_form_initial_positione(self.ball.position)
            if not self.plane_cords.is_ball_in_plane_bounds(self.ball.position):
                return -10
            elif 0<= abs_delta <= 0.175:
                return 10
            elif 0.175 < abs_delta <= 0.35:
                return 5
            # straight line between (0.35, 2.5), (0.4, 1)
            # https://www.arndt-bruenner.de/mathe/9/geradedurchzweipunkte.htm
            elif 0.35 <= abs_delta <= 0.4:
                return -30*abs_delta+ 13
            else:
                # straight line between (0.4, 1), (10, -10)
                return -55/48* abs_delta + 35/24
        
        def calculate_s_depending_on_deltaPercent(delta_percent=15):
            return 1/2 * delta_percent/100 * self.min_gravity * math.sin(self.min_angle)
        def very_strict_reward():
            """
            compares movement of the ball to a bound
            :return:
            """
            abs_delta = self.plane_cords.abs_ball_distance_form_initial_positione(self.ball.position)
            if not self.plane_cords.is_ball_in_plane_bounds(self.ball.position):
                return -10
            goal_difference = 0.005
            if 0 <= abs_delta <= goal_difference:
                return 2
            elif abs_delta <= 3 * goal_difference:
                return 1
            elif abs_delta <= 10 * goal_difference:
                return 0
            else:
                return -1

        reward = very_strict_reward()  # getBallVelocity()

        # If condition true the experiment has failed, in this case ball has left the plane.
        if not self.plane_cords.is_ball_in_plane_bounds(self.ball.position):
            self.render()
            state = self.state.copy()
            self.reset()
            # return state, reward, True, {}
            return state, reward, True, {}

        # If condition is true experiment is succesfull
        # Maybe use PlaneCoordinates to controll reward function by giving more narrow parameters

        self.counter += 1
        self.timesteps += 1

        self.performAction(action)
        # Tell Box2D to step
        self.world.Step(TIME_STEP, velocityIterations, positionIterations)
        self.world.ClearForces()
        self.description = f"debug info 0"
        done = False
        # placeholder for info
        info = {}
        self.render()
        if self.normalize:
            return self.rescaleState(), reward, done, info
        return self.state, reward, done, info

    def performAction(self, action):

        def rotate_vector(vector, angle): #angle in rad
            if radToDegree(angle) > 90:
                angle += degreeToRad(180) #so orientation is right in the end
            x, y = vector
            new_x = x * math.cos(angle) - y * math.sin(angle)
            new_y = x * math.sin(angle) + y * math.cos(angle)
            return (new_x, new_y)

        # print(f'action: in performAction  \n {action[0]}')
        # force vector in newton, point where force is applied (0,0) = center of ball, True: impuls; False: force
        rotatet_force_vector = rotate_vector((0, float(action[0])), degreeToRad(self.angle))

        self.ball.ApplyForce(rotatet_force_vector, (0, 0), False)
        # self.ball.ApplyForce(rotatet_force_vector, self.ball.worldCenter, wake=True)

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
                                    self.convertDensityToRGB(density=fixture.density, low=3.5, high=6.5,
                                                             channels=[False, True, True]),
                                    vertices)

        polygonShape.draw = my_draw_polygon

        def my_draw_circle(circle, body, fixture):
            position = body.transform * circle.pos * PPM
            position = (position[0] + SCREEN_WIDTH / 2, SCREEN_HEIGHT - position[1])
            pygame.draw.circle(self.screen, colors[body.type], [int(x) for x in position], int(circle.radius * PPM))

        circleShape.draw = my_draw_circle

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
            if demo:
                self.clock.tick(TARGET_FPS)
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
        # delete ball inclined plane etc todo:
        # Reset the reward and the counters
        self.counter = 0
        self.timesteps = 0
        self.reward = 0

        #
        self.createNewExperiment()

        # return the observation
        self.getState()
        
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
