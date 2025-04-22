from .problem_interface import ProblemInterface

class Problem_Brachistochrone(ProblemInterface):

    def __init__(self, sim):
        super().__init__(sim)
        self.variables = ["point_" + i for i in range(len(self.simulation.x_coords))]