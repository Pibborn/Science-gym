import sys
import csv
sys.path.append("/Users/lennartbaur/Documents/Arbeit/ScienceGym/Repo/science-gym")

from sciencegym.simulations.Simulation_InclinedPlane import Sim_InclinedPlane
from sciencegym.problems.Problem_InclinedPlane import Problem_InclinedPlane

from sciencegym.problems.Problem_Basketball import Problem_Basketball
from sciencegym.simulations.Simulaton_Basketball import Sim_Basketball

from sciencegym.simulations.Simulation_Brachistochrone import Sim_Brachistochrone
from sciencegym.problems.Problem_Brachistochrone import Problem_Brachistochrone

from sciencegym.simulations.Simulation_SIRV import SIRVOneTimeVaccination
from sciencegym.problems.Problem_SIRV import Problem_SIRV

from sciencegym.problems.Problem_Lagrange import Problem_Lagrange
from sciencegym.simulations.Simulation_Lagrange import Sim_Lagrange

from sciencegym.agents.StableBaselinesAgents.SACAgent import SACAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.spaces import Dict


import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor


import numpy as np



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

def train_loop(agent, train_problem, test_problem, MAX_EPISODES = 10, PRINT_EVERY = 10, VERBOSE=1, SDE=False, use_wandb = False):

    train_problem = DummyVecEnv([lambda: train_problem])
    test_problem = DummyVecEnv([lambda: test_problem])

    agent.agent = agent.create_model(train_problem, verbose=VERBOSE, use_sde=SDE)

    if use_wandb:
        agent.agent.learn(MAX_EPISODES, log_interval=PRINT_EVERY, eval_env=test_problem, eval_freq=PRINT_EVERY,
                             eval_log_path='agents/temp', callback=WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/{wandb.run.id}",
        verbose=2
    ))
    else:
        agent.agent.learn(MAX_EPISODES, log_interval=PRINT_EVERY, eval_env=test_problem, eval_freq=PRINT_EVERY,
                             eval_log_path='agents/temp')
    
    return None

def evaluate(agent, env):
    state = env.reset()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    done = False
    while not done:
        action, _ = agent.agent.predict(state, deterministic=True)
        state, reward, done, _ = env.step(action)
        terminal_state = env.buf_infos[0]["terminal_observation"]
        state = np.array(terminal_state)
        R += reward
        t += 1
        reset = t == 200
        if done or reset:
            break
    return R, state, action

def test_loop(agent, test_env, episodes, reward_threshold):
    #sirv_variables = ['mass', 'gravity', 'angle', 'force']#['susceptible', 'infected', 'recovered', 'vaccinated', 'transmission_rate', 'recovery_rate']

    if type(test_env) != DummyVecEnv:
        test_env = DummyVecEnv([lambda: test_env])

    test_rewards = []
    test_matches = 0

    succesfull_states = []

    for episode in range(episodes):
        test_reward, state, action = evaluate(agent, env=test_env)
        if test_reward >= reward_threshold:
            succesfull_states.append(state)
            test_matches += 1
        test_rewards.append(test_reward)

    
    print(f"Success rate of test episodes: {test_matches}/{episodes}={(test_matches / episodes * 100):,.2f}%")
    
    # Flatten each inner array and convert to a 2D array
    flattened_data = [arr.flatten() for arr in succesfull_states]

    # Write to CSV
    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(test_env.envs[0].variables)
        writer.writerows(flattened_data)
    
    return test_rewards

def create_env_prob(problem_string):
    if problem_string == "INCLINED":
        env = Sim_InclinedPlane()
        prob = Problem_InclinedPlane(env)
    else:
        raise ValueError(f"Problem string {problem_string} not defined!")
    
    return env, prob

if __name__ == "__main__":
    use_wandb = False
    use_monitor = False

    if use_wandb:
       # wandb.login(key="")
        wandb.init(project="my-sac-project", sync_tensorboard=True)

    train_env, train_problem = create_env_prob("INCLINED")
    test_env, test_problem = create_env_prob("INCLINED")


    input_dim, output_dim = get_env_dims(train_env)

    if use_monitor:
        train_problem = Monitor(train_problem)
        test_problem = Monitor(test_problem)

    agent = SACAgent(input_dim, output_dim, lr=1e-4, policy='MlpPolicy')

    train_loop(agent, train_problem, test_problem, MAX_EPISODES=10000, use_wandb=use_wandb)
    test_loop(agent, test_problem, episodes=1000, reward_threshold=-0.01)

    print("Finish")