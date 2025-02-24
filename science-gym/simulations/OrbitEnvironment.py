import math
from math import pi, sin, cos
from time import sleep

import gym
from cv2.cv2 import normalize
from gym.spaces import Box, Dict
import pygame
from Box2D import b2Vec2
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, edgeShape, fixtureDef)
import numpy as np
from gym.utils import seeding
from environments.EnvironmentInterface import EnvironmentInterface, rescale_movement


class Planet:
    def __init__(self, world, x, y, gravity=9.81, radius=3.0, density=5.0):
        self.world = world

        self.x, self.y = x, y
        self.gravity = gravity
        self.radius = radius
        self.density = density

        self.planet = self.create_planet()

    def create_planet(self):
        self.planet = self.world.CreateDynamicBody(
            position=(self.x, self.y),
            fixtures=fixtureDef(shape=circleShape(radius=self.radius),
                                density=self.density, friction=0.4),
            userData=None,
        )
        return self.planet

    def get_state(self):
        return np.array([self.x, self.y, self.gravity, self.radius, self.density])

    def get_world(self):
        return self.world

    def __del__(self):
        try:
            self.world.DestroyBody(self.planet)
        except:
            pass
        del self


class Satellite:
    def __init__(self, world, x, y, radius=0.3, density=0.1, force_vector=(0, 0)):
        self.world = world

        self.x, self.y = x, y
        self.radius = radius
        self.density = density
        self.force_vector = b2Vec2(force_vector[0], force_vector[1])

        self.satellite = self.create_satellite()
        pass

    def create_satellite(self):
        self.planet = self.world.CreateDynamicBody(
            position=(self.x, self.y),
            fixtures=fixtureDef(shape=circleShape(radius=self.radius),
                                density=self.density, friction=0.4),
            #linearVelocity=self.force_vector,
            userData=None,
        )
        return self.planet

    def get_state(self):
        return np.array([self.planet.position[0], self.planet.position[1], self.radius, self.density,
                         self.planet.linearVelocity[0], self.planet.linearVelocity[0]])

    def get_world(self):
        return self.world

    def __del__(self):
        try:
            self.world.DestroyBody(self.satellite)
        except:
            pass
        del self


class OrbitEnvironment(EnvironmentInterface):
    def __init__(self, seed=1000, normalize=False, rendering=False, raw_pixels=False,
                 random_planet_position=False, random_gravity=False,
                 random_satellite_position=False, random_satellite_density=False, random_satellite_size=False):
        # position of the planet randomized or in center of the screen?
        self.random_planet_position = random_planet_position
        # should the gravity of the planet be random or fixed to earth gravity (9.81 m/s^2)
        self.random_gravity = random_gravity

        # starting position of the satellite on the planet
        # if True: starting point of the satellite may be anywhere on the planet
        # if False: starting point is the top point of the planet (at 0 degrees)
        self.random_satellite_position = random_satellite_position
        # fixed or random density of the satellite
        self.random_satellite_density = random_satellite_density
        # fixed or random size of the satellite
        self.random_satellite_size = random_satellite_size

        super().__init__(seed=seed, normalize=normalize, rendering=rendering, raw_pixels=raw_pixels, gravity=0)

        self.planet = None
        self.satellite = None
        self.world.DestroyBody(self.ground)
        self.ground = None  # we don't need this here

        self.max_timesteps = 500

        if rendering:
            pygame.display.set_caption('Orbit Environment')

        # action space and observation space
        self.action_space = Box(low=np.array([-15, -15] if not self.normalize else [-1, -1]),
                                high=np.array([15, 15] if not self.normalize else [1, 1]))

        self.order = ["planet x", "planet y", "gravity", "planet radius", "planet density",
                      "satellite x", "satellite y", "satellite radius", "satellite density",
                      "force vector x", "force vector y"]
        self.ranges = {"planet x": [0, self.world_width], "planet y": [0, self.world_height], "gravity": [5., 20.],
                       "planet radius": [3., 5.], "planet density": [4., 6.], "satellite x": [0, self.world_width],
                       "satellite y": [0, self.world_height], "satellite radius": [0.2, 1.0],
                       "satellite density": [0.5, 2.], "force vector x": [-15., 15.], "force vector y": [-15., 15.]}
        self.normalized_ranges = {"planet x": [0., 1.], "planet y": [0., 1.], "gravity": [0., 1.],
                                  "planet radius": [0., 1.], "planet density": [0., 1.], "satellite x": [0., 1.],
                                  "satellite y": [0., 1.], "satellite radius": [0., 1.], "satellite density": [0., 1.],
                                  "force vector x": [-1., 1.], "force vector y": [-1., 1.]}

        observation_dict = {}
        for entry in self.order:
            if self.normalize:
                observation_dict[entry] = Box(low=self.normalized_ranges[entry][0],
                                              high=self.normalized_ranges[entry][1], shape=(1,), dtype=float)
            else:
                observation_dict[entry] = Box(low=self.ranges[entry][0],
                                              high=self.ranges[entry][1], shape=(1,), dtype=float)

        """self.observation_space = Box(low=np.array([self.ranges[entry][0] if not self.normalize
                                                   else self.normalized_ranges[entry][0] for entry in self.order]),
                                     high=np.array([self.ranges[entry][1] if not self.normalize
                                                    else self.normalized_ranges[entry][1] for entry in self.order]))"""

        self.observation_space = Dict(observation_dict)

        self.reset()

    def success(self): #todo
        return False

    def step(self, action=None):

        """
        b2Body* bk = bodies[k];
        b2Vec2 pk = bk->GetWorldCenter();
        float mk = bk->GetMass();

        b2Vec2 delta = pk - pi;
        float r = delta.Length();
        float force = G * mi * mk / (r*r);

        delta.Normalize();
        bi->ApplyForce(  force * delta, pi );
        bk->ApplyForce( -force * delta, pk );
        """
        delta = self.satellite.satellite.position - self.planet.planet.position
        r = math.sqrt(delta[0] ** 2 + delta[1] ** 2)#b2Vec2.length(delta)
        force = self.planet.gravity * self.planet.planet.mass * self.satellite.satellite.mass / (r*r)

        delta = delta / r   # normalize the vector to unit vector
        #self.planet.planet.ApplyForce( force * delta, self.planet.planet.position, False)
        # self.satellite.satellite.ApplyForce( - force * delta, self.satellite.satellite.position, False)
        self.satellite.satellite.ApplyForceToCenter(force*delta, True)
        print(force)
        """gravityDirection = self.planet.planet.position - self.satellite.satellite.position
        #gravityDirection = b2Vec2.Normalize(self.planet.planet.position - self.satellite.satellite.position)
        bodyGravity = self.satellite.satellite.mass * gravityDirection * 100000.0

        self.satellite.satellite.ApplyForce(-bodyGravity, self.satellite.satellite.position, True)"""

        return super().step(action)

    def getReward(self): #todo
        return 10

    def performAction(self, action):
        if self.normalize:
            action = rescale_movement(np.array([[-1, -1], [1, 1]]), action, np.array([[-15, -15], [15, 15]]))
        delta_x, delta_y = float(action[0]), float(action[1])
        x, y = self.satellite.satellite.position
        density = self.satellite.density
        radius = self.satellite.radius
        del self.satellite
        self.satellite = Satellite(self.world, x, y, radius=radius, density=density, force_vector=(delta_x, delta_y))
        #self.satellite = Satellite(self.world, 10, 10, radius=radius, density=density)
        self.world = self.satellite.get_world()
        # self.satellite.satellite.linearVelocity = b2Vec2(delta_x, delta_y)
        pass

    def updateState(self):
        if not (self.planet and self.satellite):  # check if planet and satellite exist in the world
            return None
        self.state = np.concatenate((np.array(self.planet.get_state()), np.array(self.satellite.get_state())))
        if self.normalize:
            self.normalized_state = self.rescaleState(np.concatenate((self.planet.get_state(), self.satellite.get_state())))
        return self.state

    def rescaleState(self, state=None):
        if state is None:
            state = self.updateState()

        if self.raw_pixels:
            self.normalized_state = rescale_movement([0, 255], self.state, [0., 1.])

        original_interval = np.array(np.array([self.normalized_ranges[entry][0] for entry in self.order]),
                                     np.array([self.normalized_ranges[entry][1] for entry in self.order]))
        to_interval = np.array(np.array([self.ranges[entry][0] for entry in self.order]),
                               np.array([self.ranges[entry][1] for entry in self.order]))
        self.normalized_state = rescale_movement(original_interval, np.array(state), to_interval)
        return self.normalized_state

    def reset(self):
        # delete old ball and basket
        if self.state is not None:
            del self.planet, self.satellite

        # reset the planet
        if self.random_planet_position:
            planet_x = self.np_random.uniform(0.3 * self.world_width, 0.7 * self.world_width)
            planet_y = self.np_random.uniform(0.3 * self.world_height, 0.7 * self.world_height)
        else:
            planet_x, planet_y = self.world_width * 0.5, self.world_height * 0.5
        planet_gravity = 500 # self.np_random.uniform(5.0, 20.) if self.random_gravity else 9.81
        planet_radius = 3.0  # self.np_random.uniform(3., 5) if self.random_planet_size else 3.0
        planet_density = 1000.0  # self.np_random.uniform(4, 6) if self.random_planet_density else 5.0
        self.planet = Planet(self.world, x=planet_x, y=planet_y, gravity=planet_gravity, radius=planet_radius,
                             density=planet_density)
        self.world = self.planet.get_world()

        # reset the satellite
        satellite_radius = self.np_random.uniform(0.2, 1.0) if self.random_satellite_size else 0.3
        starting_angle = self.np_random.uniform(0, 2 * pi) if self.random_satellite_position else 0
        satellite_x = planet_x + (planet_radius + satellite_radius) * sin(starting_angle)
        satellite_y = planet_y + (planet_radius + satellite_radius) * cos(starting_angle)

        satellite_y, satellite_x = 15, 15
        satellite_density = self.np_random.uniform(0.5, 2.0) if self.random_satellite_density else 1.0
        self.satellite = Satellite(self.world, x=satellite_x, y=satellite_y, radius=satellite_radius,
                                   density=satellite_density, force_vector=(0, 0))
        self.world = self.satellite.get_world()

        self.state = self.updateState()
        return self.rescaleState() if self.normalize else self.state

