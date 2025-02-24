from sciencegym.problems.problem_interface import ProblemInterface
from sciencegym.simulations.simulation_interface import SimulationInterface


if __name__ == '__main__':
    sim = SimulationInterface()
    prob = ProblemInterface(sim)
