import numpy as np
import pandas as pd
from sciencegym.equation import Equation

from sciencegym.simulations.simulation_interface import SimulationInterface

class ProblemInterface:

    def __init__(self, sim: SimulationInterface):
        self.simulation = sim

        self.observation_space = self.get_state_space()
        self.action_space = self.get_action_space()

    def __getattr__(self, name):
        """Delegate attribute access to instance of B if not found in A"""
        if hasattr(self.simulation, name):
            return getattr(self.simulation, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


    # Methods mirrored from SimulationInterface
    def step(self, action):
        return self.simulation.step(action)

    def get_state_space(self):
        return self.simulation.get_state_space()

    def get_current_state(self):
        return self.simulation.get_current_state()
    
    def get_action_space(self):
        return self.simulation.get_action_space()
    
    def get_simulation(self):
        return self.simulation

    # Interface methods
    def evaluation(self, candidate: Equation, data: pd.DataFrame):
        y_pred = candidate.evaluate(data)
        solution = self.solution()
        if solution is list:
            return [np.mean((single_solution.evaluate(data) - y_pred) ** 2) for single_solution in solution ]
        else:
            y_true = self.solution().evaluate(data)
            return np.mean((y_true - y_pred) ** 2)

    def validate_context(self):
        '''
        Validates the context of the problem's simulation, see Section 4.2
        of https://dl.acm.org/doi/10.1007/978-3-031-78977-9_15
        '''
        raise NotImplementedError()

    def solution(self):
        '''
        Returns the solution or solutions for the current problem.
        '''
        raise NotImplementedError()



