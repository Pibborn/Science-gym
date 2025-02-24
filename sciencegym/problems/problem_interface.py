from sciencegym.simulations.simulation_interface import SimulationInterface

class ProblemInterface:

    def __init__(self, sim: SimulationInterface):
        self.simulation = sim

    # Methods mirrored from SimulationInterface
    def step(self, action):
        return self.sim.step(action)

    def state_space(self):
        return self.sim.state_space()

    def current_state(self):
        return self.sim.current_state()

    # Interface methods
    def evaluation(self, equation):
        raise NotImplementedError()

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



