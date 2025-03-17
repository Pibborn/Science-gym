from .simulation_interface import SimulationInterface

import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from scipy.interpolate import pchip

class Sim_Brachistochrone(SimulationInterface):
    
    def __init__(self, x_start=0, x_end=np.pi, y_start=0, y_end=-2, step_per_eps=1000,
                 nb_points=10, g=9.80665, verbose=True, interactive=False,
                 testing=False, nb_test_besttimes=100):
        super().__init__()

        # compute borders -> relativ to scale
        x_margin = abs(x_end - x_start) * 0.25
        y_margin = abs(y_end - y_start) * 0.25

        y_min_plot = y_end - y_margin
        y_max_plot = y_start + y_margin

        x_min_plot = x_start - x_margin
        x_max_plot = x_end + x_margin

        ###################
        #####Constants#####
        # if test -> store more best_times and skip some steps
        self.testing = testing
        self.name = 'brachistochrone-v1'

        self.x_start = x_start
        self.y_start = y_start
        self.y_end = y_end
        self.x_end = x_end

        self.y_limit = y_min_plot

        self.step_per_eps = step_per_eps
        # agent can max go 0.1 of global scale up or down (for bigger values too unstable)
        self.max_action = abs(y_end - y_start) * 0.1
        self.verbose = verbose
        self.g = g

        self.nb_points = nb_points
        self.x_coords = np.linspace(x_start, x_end, self.nb_points + 2)
        self.sample_width = self.x_coords[1] - self.x_coords[0]
        # create a linear line from start point to end point
        self.linear_y = (lambda x: (y_end / x_end) * x)(self.x_coords)

        # R:=1 and t:=pi -> exactly half of a rotation (store R for Render function)
        self.R = 1
        self.optimal_t = self.calculate_optimal_time(1, np.pi)  # valid for start=(0, 0), end=(pi, -2)

        ##################
        ######Spaces######

        # for every sample point one action (-1:=max down, 1:=max up)
        self.action_space = Box(low=-1, high=1, shape=(self.nb_points, ))
        # observations for samples + start and end point
        # space from y_start down, also add margin (y_min_plot)
        self.observation_space = Box(low=self.y_limit, high=self.y_start, shape=(self.nb_points + 2,))
        ###################
        #####Variables#####

        # keep track of iterations + maximum steps per episode
        self.iter = 0

        self.best_y_coords = self.linear_y.copy()
        self.best_t = np.inf
        # current state -> initialize with linear line between endpoints
        self.state = self.linear_y.copy()

        # for testing: keep track of 100 best times and points
        self.best_n_ycoords = None
        self.best_n_t = None
        if self.testing:
            self.best_n_ycoords = [self.linear_y.copy() for i in range(nb_test_besttimes)]
            self.best_n_t = [np.inf for i in range(nb_test_besttimes)]

        #################
        # drawing related#
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(y_min_plot, y_max_plot)
        self.ax.set_xlim(x_min_plot, x_max_plot)
        self.ax.set_aspect("equal")
        self.ln, = self.ax.plot([], [], animated=True)
        plt.scatter([x_start, x_end], [y_start, y_end])
        plt.show(block=False)
        plt.pause(1)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ln.set_xdata(self.x_coords)
        self.ln.set_ydata([0] * len(self.x_coords))
        self.ax.draw_artist(self.ln)
        self.fig.canvas.blit(self.fig.bbox)
        self.interactive = interactive

    def step(self, action):
        # Iteration counter
        self.iter += 1
        # Flag if we are done
        done = False
        # adjust action to our environment
        # scale action -> all values between -1 and 1, according to max_action (fraction of global scale)
        action *= self.max_action
        action = np.hstack(([0], action, [0]))

        # update state
        self.old_state = self.state
        self.state += action

        # initialize reward and info string for debugging
        reward = 0
        reward_info = ""

        # Constrain action space a bit -> adjust state if out of bounds etc.
        for ix, el in enumerate(self.state):
            if ix == 0 or ix == self.nb_points + 1:
                continue
            # if action puts state out of observation space
            if el <= self.y_limit:
                self.state[ix] = self.y_end # rather: self.y_limit + 0.1 # here as before + a small unit?
                reward -= 50
                reward_info = reward_info + "\n -50\t el <= self.y_limit"
            if el >= self.y_start:
                self.state[ix] = self.y_start # same here -> add unit?
                reward -= 50
                reward_info = reward_info + "\n -50\t el >= self.y_start"

            # punish if points not falling
            if ix == 0 or ix == self.nb_points + 2:
                continue
            if el > self.state[ix - 1]:
                if action[ix] >= 0:
                    reward -= 50 / self.nb_points
                    reward_info = reward_info + "\n -50 / #points\t\t action"
                self.state[ix] = self.state[ix - 1]

        # Compute time of traversal
        traversal_time = self.calculate_traversal_time(self.state, self.sample_width, self.g)

        # Calculate reward -> only if valid points used
        if traversal_time == np.inf:
            reward -= 200
            reward_info = reward_info + "\n -200\tinvalid points"
        else:
            reward -= np.abs((self.optimal_t - traversal_time)) * 100
            reward_info = reward_info + f"\n -{round(np.abs((self.optimal_t - traversal_time)) * 100, 4)}\t traversal time"

            ## apply tautochrone property (not tested enough) -> and not reported in reward info (due to too many messages)
            for idx in range(1, len(self.state) - 1):
                curr_traversial = self.calculate_traversal_time(self.state[idx:], self.sample_width, self.g)
                # if until some point the ball won't reach the endpoint -> another big penalty
                if curr_traversial != np.inf:
                    reward -= np.abs(traversal_time - curr_traversial) * (50 / self.nb_points)
                else:
                    reward -= 50
                    reward_info = reward_info + "\n -50\tinvalid points during tautochrone"
                    break

            # Check if new record is achieved
            if traversal_time < self.best_t:
                self.best_t = traversal_time
                self.best_y_coords = self.state.copy()

        if self.iter % 50 == 0:
            self.render()
            if self.verbose and self.iter % self.step_per_eps == 0:
                print("\n==========================")
                print("optimal: \t", self.optimal_t)
                print("current: \t", traversal_time)
                print("record: \t", self.best_t)
                print("state: \t\t", self.state)
                print("action: \t", action)
                print("reward: \t", reward)
                print("\nreward summary")
                print(reward_info)
                print("==========================\n")

        if self.testing:
            if traversal_time < self.best_n_ycoords[0][0]:
                # replace the worst of 100 best times with new time
                self.best_n_ycoords[0] = self.state.copy()
                self.best_n_t[0] = traversal_time
                # sort list of tuples again (based on times which is the first element)
                self.best_n_t, self.best_n_ycoords = zip(*sorted(zip(self.best_n_t, self.best_n_ycoords), reverse=True))

        return self.state, reward, done, {}
    
    def render(self, pause_time=.01):
        self.fig.canvas.restore_region(self.bg)
        self.ln.set_data(self.x_coords, self.state)
        self.ln.set_color("b")
        self.ln.set_linestyle("-")
        self.ax.draw_artist(self.ln)

        self.ln.set_data(self.calculate_brachistochrone(self.R, self.y_end, 50)[0], np.linspace(self.y_start, self.y_end, 50))
        self.ln.set_color("r")
        self.ln.set_linestyle("--")
        self.ax.draw_artist(self.ln)

        # self.ln.set_data(self.x_coords, self.best_y_coords)
        # self.ln.set_color("g")
        # self.ln.set_linestyle("-.")
        # self.ax.draw_artist(self.ln)

        interp = pchip(self.x_coords, self.best_y_coords)
        self.ln.set_data(np.linspace(self.x_start, self.x_end, 30), interp(np.linspace(self.x_start, self.x_end, 30)))
        self.ln.set_color("g")
        self.ln.set_linestyle("-.")
        self.ax.draw_artist(self.ln)

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
        plt.pause(pause_time)

    def reset(self):
        print("reset()")
        self.state = self.linear_y.copy()

        self.iter = 0

        return self.state
    

    def calculate_optimal_time(self, R, t):
        """
        calculates the time it takes to travel until point t via radius R on a brachistochrone
        """
        g = 9.807
        return np.sqrt(R / g) * t


    def calculate_brachistochrone(self, R, y_end, points):
        x = lambda x: R * (np.arccos(x / R + 1) - np.sin(np.arccos(x / R + 1)))
        y = np.linspace(0, y_end, points)
        return x(y), y


    def calculate_traversal_time(self, y, h, g):
        """y := y-values for all sample-points
        h := width between sample-points
        g := gravity constant"""
        v = 0
        t = 0

        # same computations as in Method 1 except in a loop + some constraints
        for ys in range(1, len(y)):
            curr_y = y[ys]
            prev_y = y[ys - 1]

            #### valid point checking -> did not get good result somehow, maybe reward has to be tuned for that ####

            # boundaries to make y_points valid -> can reach the end from here
            y_min = y[-1] - (v ** 2) / (2 * g)
            y_max = (v ** 2) / (2 * g) + prev_y
            # -> if not inside the boundaries: return infinity time
            if not y_min - 0.01 <= curr_y <= y_max + 0.01:
                return np.inf

            opp = abs(curr_y - prev_y)

            adj = h
            theta = np.arctan(opp / adj)
            d = np.sqrt(opp ** 2 + adj ** 2)

            if curr_y > prev_y:
                v_f = np.sqrt(np.abs(v ** 2 - 2 * g * d * np.sin(theta)))
            else:
                v_f = np.sqrt(np.abs(v ** 2 + 2 * g * d * np.sin(theta)))
            # check if theta close to 0 -> otherwise this would result in NaN
            step_time = abs(v_f - v)/(g*np.sin(theta)) if not np.isclose(theta, 0) else adj / v_f
            t += step_time
            v = v_f

        return t

    def current_state(self):
        return self.state
    def state_space(self):
        return self.observation_space
    def action_space(self):
        return self.action_space