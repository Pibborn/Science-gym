from .problem_interface import ProblemInterface
from sciencegym.equation import Equation

class Problem_Basketball(ProblemInterface):

    def __init__(self, sim):
        super().__init__(sim)
        self.variables = ['ball_x', 'ball_y', 'angle', 'velocity', 'ball_radius', 'ball_density','ring_x', 'ring_y', 'ring_radius']

    def solution(self):
        return Equation('velocity * sin(angle) * time - 0.5 * g * time ** 2')