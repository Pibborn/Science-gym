import sys
sys.path.append("/Users/lennartbaur/Documents/Arbeit/ScienceGym/Repo/science-gym")

from sciencegym.simulations.Simulation_InclinedPlane import Sim_InclinedPlane
from sciencegym.problems.Problem_InclinedPlane import Problem_InclinedPlane
#from sciencegym.agents.StableBaselinesAgents import SACAgent

from sciencegym.agents.StableBaselinesAgents.SACAgent import SACAgent

from gym.spaces import Dict

def get_env_dims(env):
    if type(env.action_space) is not dict:
        out_dim = int(env.action_space.shape[0])
    else:
        out_dim = len(env.action_space)
    if type(env.observation_space) is not Dict:
        in_dim = env.observation_space.shape
    else:
        in_dim = len(env.observation_space)
    return in_dim, out_dim

if __name__ == "__main__":
    train_env = Sim_InclinedPlane()
    test_env = Sim_InclinedPlane()

    input_dim, output_dim = get_env_dims(train_env)

    train_problem = Problem_InclinedPlane(train_env)
    test_problem = Problem_InclinedPlane(test_env)

    agent = SACAgent(input_dim, output_dim, lr=1e-4, policy='MlpPolicy')

    agent.train_loop(train_problem, test_problem, None, verbose=2, only_testing=False)

    print("Finish")