from .simulation_interface import SimulationInterface

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
import scipy.integrate as integrate
from gym import Env
from gym.spaces import Box
from numpy.typing import NDArray


# defined constants as a shortcut to edit certain values
GRAVITAIONAL_CONSTANT = 6.672e-11   # used to calculated the force
PROPAGATION_ANGLE = 60 # angle which the 1st and 2nd body move in every step
MASS_SHIP = .1   # mass of the 3rd body, needs to be much smaller than the 2nd body

LOWER_BODY_1, UPPER_BODY_1 = 0.5, 1.0   # normalized values 
LOWER_BODY_2, UPPER_BODY_2 = 0.5, 1.0   # normalized values 
SCALE_M1, SCALE_M2 = 1e4, 1e3 # scale factor used in the internal step for the simulation 

LOWER_DIST, UPPER_DIST = 0.5, 1.0   # normalized values
SCALE_DIST = 1  # scale factor for the distance, currently set to 1 for simplicity 

LOWER_ACTION_X, UPPER_ACTION_X = 0.0, 1.0   # limits for the x-position of the 3rd body
LOWER_ACTION_Y, UPPER_ACTION_Y = 0.0, 1.0   # limits for the y-position of the 3rd body


class Body:
    """Represents a body inside the Lagrange Environment for the 3-body problem"""
    def __init__(self) -> None:
        self.mass = 0
        self.pos = [0,0]
        self.velocity = [0,0]

    def generate_mass_heavy(self) -> None:
        """Generates a random (normalized) mass for a heavier body"""
        self.mass = random.uniform(LOWER_BODY_1, UPPER_BODY_1)

    def generate_mass_light(self) -> None:
        """Generates a random (normalized) mass for a lighter body"""
        self.mass = random.uniform(LOWER_BODY_2, UPPER_BODY_2)


class Sim_Lagrange(SimulationInterface):
    def __init__(self, propagation_angle: int = PROPAGATION_ANGLE, rendering: bool = False, animate: bool = False,
                 context=0) -> None:
        """An environment for a two dimensional 3-body problem which an agent wants to learn

        Args:
            propagation_angle (optional): angle which the 1st and 2nd body move in every step
            rendering (optional): flag to determine whether to render the simulation at each step
            animate (optional): flag to determine whether the rendering show be an animation
            context (optional): 0-4 value to determine the scope of rewards
        """
        self.propagation_angle = propagation_angle
        self.rendering = rendering
        self.animate = animate
        self.context = context
        self.name = 'Lagrange-v1'

        self.bodies = [Body() for _ in range(3)]
        self.distance = 1.   # distance between the 1st and the 2nd body

        # store some of the calculated data of the step for the rendering
        self.past_results = None
        self.pos_expected = None

        # definition of action space
        # x and y position of the 3rd body that will be placed
        self.action_space = Box(    
            low  = np.array([LOWER_ACTION_X, LOWER_ACTION_Y]), 
            high = np.array([UPPER_ACTION_X, UPPER_ACTION_Y]),
            dtype=np.float32, 
        )

        # definition of the observation space
        # in order: mass of 1st and 2nd body 
        #           distance between the 1st and 2snd body          
        #           and the x and y position of the 3rd body
        # mass of 3rd body is left out since it doesn't matter
        self.observation_space = Box(
            low  = np.array([LOWER_BODY_1, LOWER_BODY_2, LOWER_DIST, LOWER_ACTION_X, LOWER_ACTION_Y]),
            high = np.array([UPPER_BODY_1, UPPER_BODY_2, UPPER_DIST, UPPER_ACTION_X, UPPER_ACTION_Y]), 
        )

    def get_state_space(self):
        return self.observation_space
    
    def get_action_space(self):
        return self.action_space
    
    def get_current_state(self):
        return np.array([self.bodies[0].mass, self.bodies[1].mass, self.distance, None, None])

    def reset(self) -> NDArray:
        """Resets the state of the environment by generating new masses and a distance for the bodies
        
        Returns:
            returns a numpy array of the new state"""
        self.bodies[0].generate_mass_heavy()
        self.bodies[1].generate_mass_light()
        self.bodies[2].mass = MASS_SHIP
        self.distance = round(random.uniform(LOWER_DIST,UPPER_DIST), ndigits=3)

        return np.array([self.bodies[0].mass, self.bodies[1].mass, self.distance, 0.0, 0.0])

    def get_reward(self, action: NDArray) -> float:
        """Calculates the reward for the reinforment learner

        Args:
            action: the positon of the 3rd position

        Returns:
            returns a float between 0 and 1"""

        # calculate the distance between the 3rd body and its expected position

        difference = float(np.linalg.norm(action - self.pos_expected))
        if self.context == 0:
            return np.exp(-difference * 10 / self.distance)
        if self.context == 1:
            return np.exp(-difference * 10 / self.distance) + np.random.normal(0, 0.2)
        if self.context == 2:
            return np.exp(-difference * 10 / self.distance) * np.random.binomial(1, 0.1)
        else:
            raise NotImplementedError('Currently, context {} is not implemented\
                                       in LagrangeEnvironment'.format(self.context))


    def differential_equation(self, _: float, x_vals: NDArray) -> NDArray:
        """simulates the movement of the bodies

        Note:
            arguments are filled in by the integrate.solve_ivp-function

        Returns:
            returns a Numpy Array with the the resulting values"""

        def force(mass_1: float, pos_1: NDArray, mass_2: float, pos_2: NDArray):
            """calculates the gravitational force between two obejcts as vector
            
            Args:
                mass_1: mass of the one body
                pos_1: position of the one body
                mass_2: mass of the other body
                pos_2: position of the other body

            Returns:
                returns a float of the gravitational force
            """
            length = np.linalg.norm(pos_2 - pos_1)
            direction = (pos_2 - pos_1) / length
            return GRAVITAIONAL_CONSTANT * mass_1 * mass_2 * direction / (length*2)

        list_of_masses = np.array([body.mass for body in self.bodies])
        list_of_positions = x_vals[:6].reshape(-1, 2)
        list_of_velocities = x_vals[6:].reshape(-1, 2)

        resulting_force: NDArray = np.empty((3, 2))
        resulting_force[:] = 0

        # apply the gravtational force on each pair of bodies
        for i, (mass_1, pos_1) in enumerate(zip(list_of_masses, list_of_positions)):
            for j, (mass_2, pos_2) in enumerate(zip(list_of_masses, list_of_positions)):
                if i == j:
                    continue
                resulting_force[i, :] += force(mass_1, pos_1, mass_2, pos_2)

        # get the acceleration for the bodies from the forces and masses
        resulting_acceleration = resulting_force / list_of_masses.reshape(3, 1).repeat(2, axis=1)
        return np.concatenate((list_of_velocities.flatten(), resulting_acceleration.flatten()))

    def internal_step(self, action: NDArray) -> None:
        """Prepares the data that is used for the simulation

        Args:
            action: the positon of the 3rd position 
        
        Returns:
            returns the (normalized) observation space"""

        # store the normalized values temporarily to revert to them after the simulation
        temp_m1 = self.bodies[0].mass
        temp_m2 = self.bodies[1].mass
        temp_dist = self.distance

        # scale the normalized values back to usable values for the simulation
        self.bodies[0].mass = round(self.bodies[0].mass * SCALE_M1)
        self.bodies[1].mass = round(self.bodies[1].mass * SCALE_M2)
        self.distance *= SCALE_DIST

        ### calculate necessary variables for the simulation ###
        # distance from center of mass to each the 1st and 2nd bodies
        r1 = self.distance * self.bodies[1].mass / (self.bodies[0].mass + self.bodies[1].mass)
        r2 = self.distance * self.bodies[0].mass / (self.bodies[0].mass + self.bodies[1].mass)

        # set the starting positon of the 1st and 2nd bodies
        self.bodies[0].pos = [-r1, 0]
        self.bodies[1].pos = [ r2, 0]
        self.bodies[2].pos = action.tolist()
        
        # calculate the velocities of 1st, 2nd and 3rd bodies
        abs_velocity_1 = np.sqrt(GRAVITAIONAL_CONSTANT * self.bodies[1].mass * r1 / (2*self.distance))
        abs_velocity_2 = np.sqrt(GRAVITAIONAL_CONSTANT * self.bodies[0].mass * r2 / (2*self.distance))
        abs_velocity_3 = float(np.linalg.norm(action)) * (abs_velocity_1 / r1)
        self.bodies[0].velocity = [0, -abs_velocity_1]
        self.bodies[1].velocity = [0,  abs_velocity_2]
        velocity_3_angle: float = np.arctan2(action[1], action[0]) + np.pi / 2 # angle of the movement of the 3rd body
        self.bodies[2].velocity: list[float] = [abs_velocity_3 * np.cos(velocity_3_angle), abs_velocity_3 * np.sin(velocity_3_angle)]

        # lists of the determined values, used later for the simulation
        list_of_posiitons = np.array([body.pos for body in self.bodies])
        list_of_velocities = np.array([body.velocity for body in self.bodies])

        time = r1 * 2 * np.pi / abs_velocity_1 # time duration for one full orbit of the 1st body
        time_sim = self.propagation_angle / 360 * time # time duration for the simulation 

        ### start simulation ###
        x_values = np.concatenate((list_of_posiitons.flatten(), list_of_velocities.flatten()))
        if self.rendering:
            results = integrate.solve_ivp(
                fun=self.differential_equation,
                y0=x_values,
                t_span=[0, time_sim],
                max_step=time_sim / 50
            )
        else:   # max_step left out to speed up the calculations 
            results = integrate.solve_ivp(
                fun=self.differential_equation,
                y0=x_values,
                t_span=[0, time_sim],
            )
        time_values = results["t"]

        # convert the output of solve_ivp to a numpy array
        # shape: [t, pos or vel, body, coordinate]
        self.past_results = results["y"].transpose().reshape(time_values.shape[0], 2, 3, 2)
        endPos_x, endPos_y = self.past_results[-1, 0, 2, :]   # end position of the 3rd body after the simulation

        # get angle to rotate the bodies back
        # needed to determine the expected position
        angle = - self.propagation_angle * np.pi / 180 
        expectedPos_x = np.cos(angle) * endPos_x - np.sin(angle) * endPos_y
        expectedPos_y = np.sin(angle) * endPos_x + np.cos(angle) * endPos_y
        self.pos_expected = np.array([expectedPos_x, expectedPos_y])

        # revert to the normalized values
        self.bodies[0].mass = temp_m1
        self.bodies[1].mass = temp_m2
        self.distance = temp_dist
        return

    def step(self, action: NDArray) -> tuple[NDArray, float, bool, dict]:
        """step-function that is used by the stable-baselines3-agent

        Args:
            action: the positon of the 3rd position 

        Returns:
            returns a tuple of the state, reward, status of the step, and additional infos"""
        
        state = self.internal_step(action)
        reward = self.get_reward(action=action)

        if self.rendering:
            self.render(action=action, reward=reward)

        self.old_state = state
        done = True
        # done = reward >= 0.95

        state = np.array([self.bodies[0].mass, self.bodies[1].mass, self.distance, action[0], action[1]])
        return state, reward, done, {"distance": self.distance}
    
    def render(self, mode="human", action=None, reward=None) -> None:
        """render function to visualize the simulation

        Args:
            mode: unused, only here for compatibility
            action (optional): the positon of the 3rd position
            reward (optional): reward of the action

        Note:
            action and reward are additional arguments, used only for dev-purposes

            when using the animation, the window is not closing after the animation is done
        """

        if self.animate:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            line, = ax.plot([],[])
            colors = "rgb"

            # store the arrays of positions of the three bodies in a separate list for ease of use
            list_of_x_vals = []
            list_of_y_vals = []
            for i in range(3):
                if self.past_results is not None:
                    x_vals, y_vals = self.past_results[:, 0, i, 0], self.past_results[:, 0, i, 1]
                    x_vals, y_vals = np.flip(x_vals), np.flip(y_vals)
                    list_of_x_vals.append(x_vals)
                    list_of_y_vals.append(y_vals)

            # plot the expected position of the 3rd body
            if self.pos_expected is not None:
                ax.plot(self.pos_expected[0], self.pos_expected[1], f"bx")

            # create the title based on the given input data
            title_string = f"m1 = {self.bodies[0].mass}, m2 = {self.bodies[1].mass}, r = {self.distance}"
            if action is not None and reward is not None:
                title_string += f"\n action = {action[0], action[1]}"
            if reward is not None: 
                title_string += f"\n reward = {reward}"
            ax.set_title(title_string)

            # store the already plotted points
            # this is used for the animate function
            plotted_x = [[] for _ in range(3)]
            plotted_y = [[] for _ in range(3)]

            def animate(i: int):
                """internal function that is used by matplotlib to animate
                
                Args:
                    i: the iteration of the animation 
                """

                # iterate over the list to plot the data of each body
                for j in range(3):
                    color = colors[j]
                    # add the new point to the list of already plotted points
                    plotted_x[j].append(list_of_x_vals[j][i])
                    plotted_y[j].append(list_of_y_vals[j][i])

                    ax.plot(plotted_x[j], plotted_y[j], color=color)

            ani = FuncAnimation(fig, animate, frames=len(list_of_x_vals[0]), interval=100, repeat=False)
            plt.show()
            return
        
        else:
            # clear axis
            plt.cla()

            colors = "rgb"
            # plot the 3 bodies
            for i in range(3):
                color = colors[i]
                if self.past_results is not None:
                    x_vals, y_vals = self.past_results[:, 0, i, 0], self.past_results[:, 0, i, 1]
                    plt.plot(x_vals, y_vals, color)
                    x_vals, y_vals = self.past_results[-1, 0, i, 0], self.past_results[-1, 0, i, 1]
                    plt.plot(x_vals, y_vals, f"{color}o")
                # plot the expected position
                if self.pos_expected is not None:
                    plt.plot(self.pos_expected[0], self.pos_expected[1], f"{color}x")
            plt.axis("equal")

            # add info about the state for context
            title_string = f"m1 = {self.bodies[0].mass}, m2 = {self.bodies[1].mass}, r = {self.distance}"
            if action is not None and reward is not None:
                title_string += f"\n action = {action[0], action[1]}"
            if reward is not None: 
                title_string += f"\n reward = {reward}"
            plt.title(title_string)

            plt.show(block=False)
            plt.pause(.1)
            return

    def seed(self, num):
        random.seed(num)