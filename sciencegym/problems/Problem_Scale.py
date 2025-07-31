from .problem_interface import ProblemInterface
import numpy as np
import pandas as pd
from sciencegym.equation import Equation

class Problem_Scale(ProblemInterface):
    def __init__(self, sim):
        super().__init__(sim)
        self.variables = ['angle', 'vel', 'position1', 'density1', 'size1', 'position2', 'density2', 'size2']

    def solution(self):
        return Equation('1 - transmission_rate / recovery_rate')