from sciencegym.problems.Problem_Basketball import Problem_Basketball
from sciencegym.problems.Problem_SIRV import Problem_SIRV
from sciencegym.simulations.Simulation_Basketball import Sim_Basketball
from sciencegym.simulations.Simulation_SIRV import SIRVOneTimeVaccination


def get_sim_and_problem(args):
    if args.simulation == 'basketball':
        sim = Sim_Basketball(
            args=args,
            seed=args.seed,
            normalize=args.normalize,
            rendering=args.rendering,
            raw_pixels=args.raw_pixels,
            random_ball_size=args.random_ball_size,
            random_density=args.random_density,
            random_basket=args.random_basket,
            random_ball_position=args.random_ball_position,
            walls=args.walls,
            context=args.context
        )
        problem = Problem_Basketball(sim=sim)
    if args.simulation == 'sirv':
        sim = SIRVOneTimeVaccination(
            args=args,
            record_training=args.rendering,
            context=args.context,
        )
        problem = Problem_SIRV(sim=sim)
    return problem, sim
