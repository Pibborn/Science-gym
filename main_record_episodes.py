import sys
import csv

from sciencegym.simulations.Simulation_InclinedPlane import Sim_InclinedPlane
from sciencegym.problems.Problem_InclinedPlane import Problem_InclinedPlane

from sciencegym.problems.Problem_Basketball import Problem_Basketball
from sciencegym.simulations.Simulaton_Basketball import Sim_Basketball

from sciencegym.simulations.Simulation_Brachistochrone import Sim_Brachistochrone
from sciencegym.problems.Problem_Brachistochrone import Problem_Brachistochrone

from sciencegym.simulations.Simulation_SIRV import SIRVOneTimeVaccination
from sciencegym.problems.Problem_SIRV import Problem_SIRV

from sciencegym.simulations.Simulation_Lagrange import Sim_Lagrange
from sciencegym.problems.Problem_Lagrange import Problem_Lagrange

from sciencegym.agents.StableBaselinesAgents.SACAgent import SACAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.spaces import Dict

import numpy as np

import wandb
from wandb.integration.sb3 import WandbCallback

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

def train_loop(agent, train_problem, test_problem, MAX_EPISODES = 10, PRINT_EVERY = 10, VERBOSE=1, SDE=False):

    train_problem = DummyVecEnv([lambda: train_problem])
    test_problem = DummyVecEnv([lambda: test_problem])

    agent.agent = agent.create_model(train_problem, verbose=VERBOSE, use_sde=SDE)
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
        print(env.buf_infos)
        state = np.array(terminal_state)
        recorded_episode = env.buf_infos[0]["record_episode"]
        R += reward
        t += 1
        reset = t == 200
        if done or reset:
            break
    return R, state, action, recorded_episode

def test_loop(agent, test_env, episodes, reward_threshold):
    #sirv_variables = ['mass', 'gravity', 'angle', 'force']#['susceptible', 'infected', 'recovered', 'vaccinated', 'transmission_rate', 'recovery_rate']

    if type(test_env) != DummyVecEnv:
        test_env = DummyVecEnv([lambda: test_env])

    test_rewards = []
    test_matches = 0

    succesfull_states = []
    succesfull_episodes = []

    for episode in range(episodes):
        test_reward, state, action, recorded_episode = evaluate(agent, env=test_env)
        if test_reward >= reward_threshold:
            succesfull_states.append(state)
            succesfull_episodes += recorded_episode
            test_matches += 1
        test_rewards.append(test_reward)

    
    print(f"Success rate of test episodes: {test_matches}/{episodes}={(test_matches / episodes * 100):,.2f}%")
    
    # Flatten each inner array and convert to a 2D array
    flattened_data = [arr.flatten() for arr in succesfull_states]
    flattened_episodes = [arr.flatten() for arr in succesfull_episodes]

    # Write to CSV
    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(test_env.envs[0].variables)
        writer.writerows(flattened_data)
    
    # Write to CSV
    with open('output_episodes.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(test_env.envs[0].variables + ["time"])
        writer.writerows(flattened_episodes)
    
    return test_rewards

if __name__ == "__main__":
    use_wandb = True
    use_monitor = False

    if use_wandb:
       # wandb.login(key="")
        wandb.init(project="science-gym", sync_tensorboard=True)

    train_env = Sim_Basketball()
    test_env = Sim_Basketball()

    input_dim, output_dim = get_env_dims(train_env)

    #train_problem = Problem_Lagrange(train_env)
    #test_problem = Problem_Lagrange(test_env)
    train_problem = Problem_Basketball(train_env)
    test_problem = Problem_Basketball(test_env)

    agent = SACAgent(input_dim, output_dim, lr=1e-4, policy='MlpPolicy')

    train_loop(agent, train_problem, test_problem, MAX_EPISODES=200)
    test_loop(agent, test_problem, episodes=200, reward_threshold=0.6)

    print("Finish")