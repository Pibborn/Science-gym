from sciencegym.simulations.simulation_interface import SimulationInterface

from abc import abstractmethod, ABC
from typing import Any

import numpy as np
from gym import Space, Env
from gym.spaces import Box
from numpy.random import Generator

from sciencegym.simulations.sirv_compartments import SIRVVitalDynamics

class SIRVEnvironment(SimulationInterface, ABC):
    def __init__(self, death_rate, birth_rate, population: int):
        self.name = 'SIRV-v1'

        # observation space consists of the four compartments and transmission/recovery rates
        self.observation_space: Space = Box(low=0, high=1, shape=[6])

        self.action_space: Space = Box(low=0, high=1, shape=[1])

        self.death_rate = death_rate
        self.birth_rate = birth_rate

        # variables for the simulation in each episode
        self.SIRV_model: SIRVVitalDynamics | None = None
        self.last_observation: tuple = (None, None, None, None)

        # initial compartments
        self.initial_infected = None
        self.initial_recovered = None
        self.initial_vaccinated = None
        self.initial_susceptible = None

        self.population = population

        self.episode_number = 0
        self.episode_steps = 0
    
    def get_current_state(self):
        return self.get_observation()
    
    def get_state_space(self):
        return self.observation_space
    
    def get_action_space(self):
        return self.action_space

    def seed(self, num):
        self.np_random: Generator = np.random.default_rng(seed=num)

    def step(self, action: np.ndarray):
        # apply the action to the simulation
        self.apply_action(action)

        # get the state of the simulation after applying the action
        observation = self.get_observation()
        self.old_state = observation

        # calculate the reward
        reward = self.get_reward()

        # determine if the episode is finished
        done = self.is_done()

        info = {}

        # save the last observation
        self.last_observation = observation

        self.episode_steps += 1
        self.episode_number += 0 if done is False else 1

        return observation, reward, done, info

    @abstractmethod
    def get_reward(self):
        pass

    @abstractmethod
    def apply_action(self, action):
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def render(self, mode="human"):
        pass

    def get_observation(self):
        compartments = self.SIRV_model.get_compartments()
        # norm the compartments, so they fit in [0, 1]
        normed_compartments = tuple(compartment / self.SIRV_model.get_full_population() for compartment in compartments)
        # append transmission+recovery rate
        observation = (*normed_compartments, self.SIRV_model.transmission_rate, self.SIRV_model.recovery_rate)
        return observation
    

    @staticmethod
    def run_simulation(sim: SIRVVitalDynamics, max_steps: int):
        """Runs the provided simulation for up to max_steps"""
        done = False
        observations = []
        while not done:
            last_observation = sim.get_compartments()
            observations.append(last_observation)

            sim.step()

            new_observation = sim.get_compartments()

            if sim.infected < 0.8:  # infection burnt out
                done = True
            elif sim.susceptible < 0.8:  # susceptible compartment exhausted
                done = True
            # FIXME: np.isclose is quite expensive
            elif np.all(np.isclose(new_observation, last_observation)):  # in steady state
                done = True
            else:
                done = sim.steps >= max_steps
        return done, sim.steps, observations

    def reset(self):
        self.episode_steps = 0

        initial_population = self.population

        # chance of disease transmission on contact
        transmission_chance = self.np_random.random()

        # average nr of contacts per person
        average_contacts = self.np_random.integers(low=1, high=initial_population)  # per step

        # nr of contacts which result in infection per infectious person (assuming all contacted are susceptible)
        transmissions = transmission_chance * average_contacts

        # calculate transmission rate, bounded between self.death_rate and 1 (with a minimum of 0.01)
        transmission_rate = np.clip(self.death_rate + (1 - self.death_rate) * transmissions / initial_population,
                                    a_min=0.01, a_max=1.)

        # ensure r_0 > 1
        recovery_rate = np.clip((transmission_rate - self.death_rate) * self.np_random.random(),
                                a_min=0.001, a_max=1.)

        # initial compartment states
        self.initial_infected = 1
        self.initial_recovered = 0
        self.initial_vaccinated = 0
        self.initial_susceptible = (initial_population - self.initial_infected
                                    - self.initial_recovered - self.initial_vaccinated)

        # initialize simulation for the episode
        self.SIRV_model = SIRVVitalDynamics(
            transmission_rate=transmission_rate,
            recovery_rate=recovery_rate,
            vaccination_rate=0,
            susceptible=self.initial_susceptible,
            infected=self.initial_infected,
            recovered=self.initial_recovered,
            vaccinated=self.initial_vaccinated,
            birth_rate=self.birth_rate,
            death_rate=self.death_rate
        )

        # get initial observation
        self.last_observation = self.get_observation()

        return self.last_observation


class SIRVOneTimeVaccination(SIRVEnvironment):
    def __init__(self, args, population=10_000, record_training: bool = False, recording_interval: int = 150,
                 context: str = 'classic'):
        super().__init__(0, 0, population)
        self.args = args
        self.visualize_training: bool = record_training
        self.recorded_episodes: list = []
        self.context = context
        self.recording_interval = recording_interval  # how many steps there should be between recordings

    def apply_action(self, action: np.ndarray):
        # move a proportion of individuals into the vaccinated compartment
        self.SIRV_model.vaccinated = self.SIRV_model.susceptible * action[0]
        self.SIRV_model.susceptible -= self.SIRV_model.susceptible * action[0]

        # adjust saved initial conditions
        self.initial_susceptible = self.SIRV_model.susceptible
        self.initial_vaccinated = self.SIRV_model.vaccinated

        # do full simulation for visualisation purposes
        if self.visualize_training is True and self.episode_number % self.recording_interval == 0:
            self.render()

        # simulate a step
        self.SIRV_model.step()

    def get_reward(self):
        # calculate reward according to distance from one-to-one infection rate
        if self.context == 'classic':
            return -abs(self.SIRV_model.infected - self.initial_infected) / self.initial_infected * 100
        elif self.context == 'noise':
            return (-abs(self.SIRV_model.infected - self.initial_infected) / self.initial_infected * 100 +
                    np.random.normal(self.args.noise_loc, self.args.noise_scale))
        elif self.context == 'sparse':
            return (-abs(self.SIRV_model.infected - self.initial_infected) / self.initial_infected * 100) >= self.args.sparse_thr
        else:
            raise NotImplementedError(f"Context {self.context} not implemented")



    def is_done(self) -> bool:
        return True

    def render(self, mode="human"):
        timesteps = 10_000
        test_sim = SIRVVitalDynamics(
            transmission_rate=self.SIRV_model.transmission_rate,
            recovery_rate=self.SIRV_model.recovery_rate,
            vaccination_rate=0,
            susceptible=self.SIRV_model.susceptible,
            infected=self.SIRV_model.infected,
            recovered=self.SIRV_model.recovered,
            vaccinated=self.SIRV_model.vaccinated,
            birth_rate=self.birth_rate,
            death_rate=self.death_rate
        )

        _, _, compartments = self.run_simulation(test_sim, timesteps)

        n = self.SIRV_model.get_full_population()
        data = [[s / n * 10_000 for s, i, r, v in compartments],
                [i / n * 10_000 for s, i, r, v in compartments],
                [r / n * 10_000 for s, i, r, v in compartments],
                [v / n * 10_000 for s, i, r, v in compartments]]

        self.recorded_episodes.append(data)