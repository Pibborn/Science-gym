import numpy as np
from gym import spaces
import gym

class SimulationInterface(gym.Env):

    def __init__(self):
        pass

    def step(self, action):
        raise NotImplementedError()

    def current_state(self) -> gym.spaces.Space:
        '''
        Returns an array representing the current status of the simulation.
        '''
        raise NotImplementedError()

    def state_space(self) -> gym.spaces.Space:
        '''
        Returns a gymnasium.spaces.Space containing the ranges of the state space.
        '''
        return NotImplementedError()

    def action_space(self) -> gym.spaces.Space:
        '''
        Return a gymnasium.spaces.Space
        '''
        return NotImplementedError()

    def supports_context(self) -> list:
        return NotImplementedError()

    def reset(self) -> tuple[gym.spaces.Space, dict]:
        '''
        Resets the simulation to the initial conditions and returns the initial
        conditions of the new environment, alongside additional information.
        '''
        raise NotImplementedError()




