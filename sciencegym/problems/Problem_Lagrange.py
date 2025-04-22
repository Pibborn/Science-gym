from .problem_interface import ProblemInterface
from sciencegym.equation import Equation

class Problem_Lagrange(ProblemInterface):
    def __init__(self, sim):
        super().__init__(sim)
        self.variables = ['body_1_mass', 'body_2_mass', 'distance_b1_b2', 
                          'bod_3_posX', 'bod_3_posY']

    def solution(self):
        return [Equation('3**(1/2) / 2 * distance_b1_b2'), Equation('distance_b1_b2 / 2 - d')]

    