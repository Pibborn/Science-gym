from .simulation_interface import SimulationInterface
import random
import numpy as np

import Box2D
from Box2D import b2Color, b2Vec2, b2DrawExtended

import gym
from gym import spaces
from gym.spaces import Box, Dict

# from ScaleEnvironment.framework import (Framework, Keys, main)
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, edgeShape, fixtureDef)

import math

import pygame

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


class Sim_InclinedPlane(SimulationInterface):

    def __init__(self):
        super().__init__()

        self.order = None
        self.raw_pixels = False
        self.normalize = False

        # Pygame setup
        self.rendering = True
        if self.rendering:
            pygame.init()

            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('Inclined Plane')
        else:
            self.screen = pygame.display.set_mode((1, 1))
        self.clock = pygame.time.Clock()

        # internal simulation information
        self.counter = 0
        self.timesteps = 0
        self.reward = 0
        
        self.gs_list = [-5, -6, -7, -8, -9] #, gravity_venus, -5, -5.5, -6.5, -7.5, -8.5, -9.5 ,-6, -7, -9, -10] # , gravity_mars, gravity_moon]
        self.min_gravity = min(self.gs_list)
        self.max_gravity= max(self.gs_list)
        self.gravity = None

        # weights
        self.min_mass = 5
        self.max_mass = 200

        # angles
        self.min_angle = 5
        self.max_angle = 70

        # forces
        min_force = 0
        self.max_force = abs(self.max_mass * self.max_gravity) * math.sin(degreeToRad(self.max_angle))

        # Box2d world setup
        # Create the world
        self.world = world(gravity=(0, -9.80665), doSleep=True)
        
        # Create the action space of the Sim
        self.action_space = Box(
            low=min_force,
            high=self.max_force,
            shape=(1,), dtype=np.float32)
        
        # Create the observation Sapce of the Sim
        observation_dict = {
                "mass": Box(low=self.min_mass, high=self.max_mass, shape=(1,), dtype=np.float32),
                "gravity": Box(low=self.min_gravity, high=self.max_gravity, shape=(1,), dtype=np.float32),
                "angle": Box(low=self.min_angle, high=self.max_angle, shape=(1,), dtype=np.float32),
                "force": Box(low=min_force, high=self.max_force, shape=(1,), dtype=np.float32)
            }
        
        self.observation_space = spaces.Dict(spaces=observation_dict)  # convert to Spaces Dict

        self.observation_space = self.convert_observation_space(self.observation_space)

        self.force = 0 

        self.reset()

        # state calculation
        self.internal_state = None
        self.state = self.getState()
        self.normalized_state = None

        return None

    def convert_observation_space(self, obs_space: spaces.Dict, order: list[str] = []) -> spaces.Box:
        """Convert an existing scale environment so that the observation space is a Box instead of a Dict.

                :type obs_space: gym.spaces.Dict
                :param obs_space: the given observation space in the old format
                :type order: list[str]
                :param order: give a list of the descriptions as an order for the new observation space
                :returns: the new observation space as a Box
                :rtype: gym.spaces.Box
        """
        # not the best solution
        if not order:
            low = [obs_space[x].low[0] for x in obs_space]
            high = [obs_space[x].high[0] for x in obs_space]
            # shape = obs_space[list(obs_space)[0]].shape[2]
        else:
            low = [obs_space[entry].low[0] for entry in order]
            high = [obs_space[entry].high[0] for entry in order]
        observation_space = spaces.Box(low=np.array(low), high=np.array(high),
                                       # shape=(shape,), # todo: fix shape
                                       dtype=np.float32)
        return observation_space

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
            fixtures=fixtureDef(shape=polygonShape(vertices=coordinates[0:3])),
            userData=color)

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

    def step(self, action):
        #print("#############\nStep is called")
        #return self.step_no_sim(action)
        return self.simulated_step(action)

    def step_no_sim(self, action):
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
        #print(f"Force of agent{Fagent}")
        degangle = self.angle
        radangle = degreeToRad(degangle)
        gravity = self.gravity #round(random.uniform(6, 12), 2) #self.gravity #round(random.uniform(6, 12), 2) # 9.80665
        mass = self.mass

        Fgoal = mass * gravity * math.sin(radangle)

        error = Fagent + Fgoal  # Net force
        reward = - (error / self.max_force) ** 2

        # Optional precision bonus
        if abs(error) < 0.1:
            reward += 2


        # accgoal = gravity * math.sin(radangle) * delta_percent
        
        # vgoal = accgoal * t
        
        # sgoal = 1/2 * accgoal * t**2

        # Fres = abs(Fgoal - Fagent)
        # accagent = Fres/mass # a
        # vagent = accagent * t
        # sagent = 1/2 * accagent * t**2

        # isstate = sagent
        # goalstate = sgoal
        # if 0 <= isstate <= goalstate:
        #     reward = 2
        # elif isstate <= 3*goalstate:
        #     reward = 1
        # elif isstate <= 10*goalstate:
        #     reward = 0
        # else:
        #     reward = -1

        # reward = -abs(sgoal - sagent)  # Negative distance


        self.old_state = self.state
        #self.gravity = round(random.uniform(6, 12), 2) 
        state = np.array([mass, gravity, radangle, Fagent], dtype=np.float32)
        #print(f"state in step: {state}")
        self.state = state
        return state, reward, True, {}
    
    # Simulated Step
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

        timesteps = 10

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
        #self.reset()
        done = True
        return state, reward, done, info

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
           # self.reset()
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
           # if demo:
            if True:
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

    # simulated step end
    def reset(self):
        #print("#############\nReset is called")
        self.counter = 0
        self.timesteps = 0
        self.reward = 0

        self.createNewExperiment()

        self.getState()
       
        return self.state
    
    def get_current_state(self):
        #self.state = np.array([self.ball.mass, self.gravity, np.deg2rad(self.angle), self.force], dtype=np.float32)
        return self.state if self.state else self.getState()

    def get_action_space(self):
        return self.action_space
    
    def get_state_space(self):
        return self.observation_space
    