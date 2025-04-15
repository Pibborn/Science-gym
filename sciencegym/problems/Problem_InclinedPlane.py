from .problem_interface import ProblemInterface
from sciencegym.equation import Equation

class Problem_InclinedPlane(ProblemInterface):

    def __init__(self, sim):
        super().__init__(sim)
        self.variables = ['mass','gravity','angle','force']
        
        def solution(self):
            return Equation('mass * gravity * sin(angle)')