from time import sleep

import gym
from gym.spaces import Box
import pygame
from Box2D import b2Vec2
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, edgeShape, fixtureDef)
import numpy as np
from gym.utils import seeding
from environments.EnvironmentInterface import EnvironmentInterface, rescale_movement


class Ball:
    def __init__(self, world, x, y, angle, velocity, radius=1.0, density=1.0, delta_x=0, delta_y=0):
        self.world = world

        self.x, self.y = x, y
        self.angle = angle
        self.velocity = velocity
        self.radius = radius
        self.density = density

        self.delta_x, self.delta_y = delta_x, delta_y

        self.ball = self.create_ball()
        pass

    def create_ball(self):
        self.ball = self.world.CreateDynamicBody(
            position=(self.x, self.y),
            fixtures=fixtureDef(shape=circleShape(radius=self.radius),
                                density=self.density, friction=0.4,
                                restitution=0.6),  # restitution makes the ball bouncy
            linearVelocity=b2Vec2(self.delta_x, self.delta_y),
            userData=None,
        )
        return self.ball

    def get_state(self):
        position1 = self.ball.position[0]
        position2 = self.ball.position[1]
        angle = self.ball.angle
        velocity = float(self.ball.angularVelocity)
        return np.array([position1, position2, angle, velocity, self.radius, self.density])

    def get_world(self):
        return self.world

    def __del__(self):
        try:
            self.world.DestroyBody(self.ball)
        except:
            pass
        del self


class Basket:
    def __init__(self, world, x=38, y=20, radius=1.5, width=0.2):
        self.world = world
        self.x = x
        self.y = y
        self.radius = radius
        self.width = width

        self.basket = self.create_basket()

    def _create_vertices(self):
        x1, y1 = self.x - self.radius - self.width, self.y
        x2, y2 = x1, self.y - 2 * self.radius - self.width
        x3, y3 = self.x + self.radius + self.width, y2
        x4, y4 = x3, self.y + 2 * self.radius
        x5, y5 = self.x + self.radius, y4
        x6, y6 = x5, self.y - 2 * self.radius
        x7, y7 = self.x - self.radius, y6
        x8, y8 = x7, y1
        vertices = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7), (x8, y8)]
        return vertices

    def create_basket(self):
        p1, p2, p3, p4, p5, p6, p7, p8 = self._create_vertices()
        rectangle1 = [p1, p2, p7, p8]
        rectangle2 = [p2, p3, p6, p7]
        rectangle3 = [p3, p4, p5, p6]
        self.basket = self.world.CreateStaticBody(
            position=(0, 0),
            # fixtures=fixtureDef(shape=polygonShape(vertices=self._create_vertices()), density=100),
            fixtures=[fixtureDef(shape=polygonShape(vertices=rectangle1), density=100),
                      fixtureDef(shape=polygonShape(vertices=rectangle2), density=100),
                      fixtureDef(shape=polygonShape(vertices=rectangle3), density=100)],
            userData=None,
        )
        return self.basket

    def get_state(self):
        return np.array([self.x, self.y, self.radius])

    def get_world(self):
        return self.world

    def __del__(self):
        self.world.DestroyBody(self.basket)
        del self


class BasketballEnvironment(EnvironmentInterface):
    def __init__(self, seed=None, normalize=False, rendering=True, raw_pixels=False, random_ball_size=True,
                 random_density=False, random_basket=False, random_ball_position=True, walls=0):
        # should the size of the ball be fixed or random?
        self.random_ball_size = random_ball_size
        # should the density of the ball be fixed or random?
        self.random_density = random_density
        # should the basket be placed created on the right side?
        self.random_basket = random_basket
        # start location of the basketball random or not?
        self.random_ball_position = random_ball_position

        # create Environment with interface
        super().__init__(seed=seed, normalize=normalize, rendering=rendering, raw_pixels=raw_pixels, walls=walls)

        self.ball = None
        self.basket = None
        self.touched_the_basket = False  # for reward determination later
        self.starting_position = None

        self.max_timesteps = 300

        if rendering:
            pygame.display.set_caption('Basketball Environment')

        # action space and observation space
        self.action_space = Box(low=np.array([0, 0]), high=np.array([15, 15] if not self.normalize else [1, 1]))
        # alternatively: use (total) velocity and angle to throw the ball
        # self.action_space = gym.spaces.Box(low=np.array([-np.pi, 0]), high=np.array([np.pi, 100]))

        # first, we have the ball information: x coordinate, y coordinate, angle, velocity, radius/size, density
        # then, we have the basket info: x coordinate, y coordinate (both of the center of the ring), radius of the ring
        self.observation_space = Box(low=np.array([0, 0, - np.pi, -10, 0.5, 4, 0, 0, 0.5] if not self.normalize
                                                  else [0, 0, -1, -1, 0, 0, 0, 0, 0]),
                                     high=np.array([self.world_width, self.world_height, np.pi, 10, 1.5, 6,
                                                    self.world_width, self.world_height, 3] if not self.normalize
                                                   else [1 for _ in range(9)]))
        self.observation_space = Box(low=np.array([0, 0, - np.pi, -10, 0.5, 4, 0, 0, 0.5] if not self.normalize
                                                  else [0, 0, -1, -1, 0, 0, 0, 0, 0]),
                                     high=np.array([self.world_width, self.world_height, np.pi, 10, 1.5, 6,
                                                    self.world_width, self.world_height, 3] if not self.normalize
                                                   else [1 for _ in range(9)]))

        self.reset()

    def success(self):
        # idea: check, if box x coordinates in the range of the position of the basket
        # and if the ball is completely under the line
        # self.state = self.updateState()
        ball_x, ball_y = self.ball.ball.position

        if (self.basket.x - self.basket.radius <= ball_x - self.ball.radius
                and self.basket.x + self.basket.radius >= ball_x + self.ball.radius
                and self.basket.y >= ball_y + self.ball.radius
                and self.basket.y - 2 * self.basket.radius <= ball_y - self.ball.radius):
            return True
        return False

    def getReward(self):
        # ball in net --> stop and give maximum reward
        if self.success():
            return 100
        # ball either touched the basket right now or before --> reward = 10
        elif len(self.basket.basket.contacts) > 0 or self.touched_the_basket:
            if not self.touched_the_basket:
                self.touched_the_basket = True
            return 10
        # ball outside of the field --> reward should be negative
        elif self.testFailed():
            return -10
        else:
            return 0

    def testFailed(self):
        ball_x, ball_y = self.updateState()[0:2]
        # check if ball is outside the displayed area
        if not 0 <= ball_x <= self.world_width:
            return True
        return False

    def printSuccess(self):
        decimal_places = 2
        distance_to_basket = self.basket.x - self.basket.width - self.basket.radius - self.starting_position[0]
        return_message = f"Success!\t " \
                         f"| Starting position: {[float(f'%.{decimal_places}f' % n) for n in self.starting_position]}\t" \
                         f"| Distance: {f'%.{decimal_places}f' % distance_to_basket}\t  " \
                         f"| Mass: {f'%.{decimal_places}f' % self.ball.ball.mass} "
        return return_message

    def performAction(self, action):
        if self.normalize:
            action = rescale_movement(np.array([[0, 0], [1, 1]]), action, np.array([[0, 0], [15, 15]]))
        delta_x, delta_y = float(action[0]), float(action[1])
        x, y = self.ball.ball.position
        density = self.ball.density
        radius = self.ball.radius
        del self.ball
        self.ball = Ball(self.world, x=x, y=y, angle=0, velocity=0, radius=radius,
                         density=density)  # , delta_x=delta_x, delta_y=delta_y)
        self.world = self.ball.get_world()
        self.ball.ball.linearVelocity = b2Vec2(delta_x, delta_y)
        pass

    def updateState(self):
        if not (self.ball and self.basket):  # check if basket and ball exist in the world
            return None
        self.state = np.concatenate((self.ball.get_state(), self.basket.get_state()))
        if self.normalize:
            self.normalized_state = self.rescaleState(np.concatenate((self.ball.get_state(), self.basket.get_state())))
        return self.state

    def rescaleState(self, state=None):
        if state is None:
            state = self.updateState()
        if self.raw_pixels:
            self.normalized_state = rescale_movement([0, 255], self.state, [0., 1.])

        else:
            self.normalized_state = rescale_movement(
                # [self.observation_space.low, self.observation_space.high],
                np.array([[0, 0, - np.pi, -10, 0.5, 4, 0, 0, 0.5],
                          [self.world_width, self.world_height, np.pi, 10, 1.5, 6,
                           self.world_width, self.world_height, 3]]),
                state,
                np.array([np.array([0, 0, -1, -1, 0, 0, 0, 0, 0]), np.array([1 for _ in range(9)])]))
        return self.normalized_state

    def reset(self):
        # delete old ball and basket
        if self.state is not None:
            del self.ball, self.basket

        # reset the ball
        if self.random_ball_position:
            ball_x = self.np_random.uniform(0.1 * self.world_width, 0.5 * self.world_width)
            ball_y = self.np_random.uniform(0.4 * self.world_height, 0.7 * self.world_height)
        else:
            ball_x, ball_y = self.world_width * 0.4, self.world_height * 0.6
        ball_radius = self.np_random.uniform(0.5, 1.5) if self.random_ball_size else 1.0
        ball_density = self.np_random.uniform(4, 6) if self.random_density else 5.0
        self.ball = Ball(self.world, x=ball_x, y=ball_y, angle=0, velocity=0, radius=ball_radius, density=ball_density)
        self.world = self.ball.get_world()

        # reset the basket
        if self.random_basket:
            basket_y = self.np_random.uniform(0.2 * self.world_height, 0.8 * self.world_height)
            basket_radius = self.np_random.uniform(ball_radius, 2 * ball_radius)
            basket_width = self.np_random.uniform(0.1, 1.0)
            basket_x = self.world_width - basket_width - basket_radius
        else:
            basket_radius, basket_width = 1.8, 0.4
            basket_y, basket_x = 0.6 * self.world_height, self.world_width - basket_width - basket_radius
        self.basket = Basket(self.world, x=basket_x, y=basket_y, radius=basket_radius, width=basket_width)
        self.world = self.basket.get_world()

        self.starting_position = (ball_x, ball_y)
        self.touched_the_basket = False
        self.state = self.updateState()
        return self.rescaleState() if self.normalize else self.state
