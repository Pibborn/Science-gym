from .problem_interface import ProblemInterface
import numpy as np
import pandas as pd
from sciencegym.equation import Equation

class Problem_DropFriction(ProblemInterface):
    def __init__(self, sim):
        super().__init__(sim)
        self.variables = ['drop_length', 'adv', 'rec', 'avg_vel', 'width']

    def solution(self):
        return Equation('0')
