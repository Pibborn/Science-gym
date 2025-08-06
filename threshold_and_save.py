import csv
from datetime import datetime
from typing import Dict

import numpy as np

from sciencegym.agents.StableBaselinesAgents.SACAgent import SACAgent
from sciencegym.agents.StableBaselinesAgents.A2CAgent import A2CAgent
from sciencegym.config.parse_arguments import parse_arguments
from sciencegym.equation_discovery.symbolic_regression import run_symbolic_regression
from sciencegym.problems.Problem_DropFriction import Problem_DropFriction
from sciencegym.problems.get_sim_and_problem import get_sim_and_problem
from sciencegym.simulations.Simulation_DropFriction import Sim_DropFriction
from sciencegym.simulations.Simulation_Lagrange import Sim_Lagrange
from sciencegym.simulations.Simulation_InclinedPlane import Sim_InclinedPlane
from sciencegym.problems.Problem_Lagrange import Problem_Lagrange
from sciencegym.problems.Problem_InclinedPlane import Problem_InclinedPlane

from stable_baselines3.common.vec_env import DummyVecEnv
from gym.spaces import Dict as GymDict

import time

from sciencegym.utils.utils import save_arguments, save_results, get_exsisting_csv_path



def get_env_dims(env):
    if not isinstance(env.action_space, GymDict):
        out_dim = int(env.action_space.shape[0])
    else:
        out_dim = len(env.action_space)
    if not isinstance(env.observation_space, GymDict):
        in_dim = env.observation_space.shape
    else:
        in_dim = len(env.observation_space)
    return in_dim, out_dim


def train_agent(args, sim, problem):
    """Return a trained SACAgent on `sim`."""
    vec_env = DummyVecEnv([lambda: problem])
    in_dim, out_dim = get_env_dims(sim)
    if args.rl_agent == 'SAC':
        agent = SACAgent(in_dim, out_dim,
                         lr=args.lr,
                         policy=args.policy)
    elif args.rl_agent == 'A2C':
        agent = A2CAgent(in_dim, out_dim, lr=args.lr,
                         policy=args.policy
                         )
    else:
        raise NotImplementedError(f"Unrecognized RL agent: {args.rl_agent}")
    model = agent.create_model(vec_env, verbose=args.verbose)
    model.learn(args.rl_train_steps, log_interval=1000)
    agent.agent = model
    return agent, problem


def record_successful_episodes(args, agent, problem, csv_path, threshold):
    """Roll out evaluation episodes, store those above reward threshold."""
    print('Evaluating...')
    vec_env = DummyVecEnv([lambda: problem])
    states = []
    episodes = []
    possible_states = 0
    reward_sum_all_episodes = 0
    num_full_episodes = 0

    for _ in range(args.rl_test_episodes):
        obs = vec_env.reset()
        done, reward_sum, t = False, 0.0, 0
        while not done:
            if isinstance(vec_env.envs[0], Problem_DropFriction):
                action, _ = agent.agent.predict(obs)
            else:
                action, _ = agent.agent.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            t += 1
            reward_sum += reward
            if isinstance(problem, Problem_DropFriction):
                possible_states += 1
                if reward >= threshold:
                    states.append(obs)
                if done or t == args.t_end_test:
                    break
            else:
                if done or t == args.t_end_test:
                    possible_states += 1
                    terminal_obs = vec_env.buf_infos[0]["terminal_observation"]
                    episode = vec_env.buf_infos[0].get("record_episode", np.array([np.nan]))
                    if reward_sum >= threshold:
                        states.append(terminal_obs)
                        episodes += episode
                    break
        reward_sum_all_episodes += reward_sum
        num_full_episodes += 1

    if len(states) == 0:
        raise RuntimeError(f"No successful episodes found: nothing to record."
                           f" You can try to increase the rl_test_episodes or "
                           f"lower the success_thr")

    # flatten and write
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(problem.variables)
        writer.writerows([np.array(s).flatten() for s in states])

    with open(csv_path.with_name(csv_path.stem + "_episodes.csv"), "w",
              newline="") as f:
        writer = csv.writer(f)
        writer.writerow(vec_env.envs[0].variables + ["time"])
        writer.writerows([e.flatten() for e in episodes])
    record_dict = {
        '#_records_saved': len(states),
        '#_possible_states': possible_states,
        'SuccessRate': len(states) / possible_states,
        'avg_reward': float((reward_sum_all_episodes / num_full_episodes)[0])
    }
    print(f"Saved {len(states)} successful trajectories to {csv_path}\n"
          f"from {possible_states} which is a ratio of {len(states) / possible_states}")

    return record_dict


def main():
    args = parse_arguments()

    problem, sim = get_sim_and_problem(args)

    args.result_dir = (args.root_dir / args.result_dir / args.exp_name
                       / args.simulation / args.rl_agent / args.equation_discoverer
                       / args.context / f"seed_{args.seed}"
                       / f"{datetime.now().strftime('%Y_%b_%d_%H_%M')}")
    save_arguments(args)
    print(f"\n=== Training {args.simulation} (context={args.context}) ===")
    start_time = time.time()
    records_dict = {}
    if args.regress_only:
        csv_path = get_exsisting_csv_path(args)
        end_time_rl = time.time()
    else:
        print('Training RL agent...')
        csv_path = args.result_dir / "successful_states.csv"
        agent, problem = train_agent(args, sim, problem)
        end_time_rl = time.time()

        records_dict = record_successful_episodes(
            args,
            agent,
            problem,
            csv_path=csv_path,
            threshold=args.success_thr
        )

    results = run_symbolic_regression(args, csv_path, problem)
    end_time_sr = time.time()

    results['overview']['time_rl'] = end_time_rl - start_time
    results['overview']['time_sr'] = end_time_sr - end_time_rl
    results['overview']['#_record_dict'] = records_dict
    results['overview']['done'] = True

    save_results(args, results)


if __name__ == "__main__":
    main()
