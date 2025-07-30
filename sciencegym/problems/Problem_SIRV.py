from .problem_interface import ProblemInterface
import numpy as np
import pandas as pd
from sciencegym.equation import Equation

class Problem_SIRV(ProblemInterface):
    def __init__(self, sim):
        super().__init__(sim)
        self.variables = ['susceptible', 'infected', 'recovered', 
                          'vaccinated', 'transmission_rate', 'recovery_rate']

    def solution(self):
        return Equation('1 - recovery_rate / transmission_rate ')

    #def evaluate(self, candidate: Equation, data: pd.DataFrame):
    #    y_true = self.solution().evaluate(data)
    #    y_pred = candidate.evaluate(data)
    #    return np.mean((y_true - y_pred) ** 2)