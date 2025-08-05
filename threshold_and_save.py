import json
import os
import csv
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from tqdm import tqdm

from sciencegym.agents.StableBaselinesAgents.PPOAgent import PPOAgent
from sciencegym.agents.StableBaselinesAgents.SACAgent import SACAgent
from sciencegym.agents.StableBaselinesAgents.A2CAgent import A2CAgent
from sciencegym.config.config_basketball import ConfigBasketball
from sciencegym.config.config_general import ConfigGeneral
from sciencegym.config.config_gplearn import ConfigGPlearn
from sciencegym.config.config_pysr import ConfigPySR
from sciencegym.config.config_rl_agent import ConfigRLAgent
from sciencegym.problems.Problem_DropFriction import Problem_DropFriction
from sciencegym.simulations.Simulation_Basketball import Sim_Basketball
from sciencegym.simulations.Simulation_DropFriction import Sim_DropFriction
from sciencegym.simulations.Simulation_SIRV import SIRVOneTimeVaccination
from sciencegym.simulations.Simulation_Lagrange import Sim_Lagrange
from sciencegym.simulations.Simulation_InclinedPlane import Sim_InclinedPlane
from sciencegym.problems.Problem_Basketball import Problem_Basketball
from sciencegym.problems.Problem_SIRV import Problem_SIRV
from sciencegym.problems.Problem_Lagrange import Problem_Lagrange
from sciencegym.problems.Problem_InclinedPlane import Problem_InclinedPlane

from stable_baselines3.common.vec_env import DummyVecEnv
from gym.spaces import Dict as GymDict

from pysr import PySRRegressor
from sciencegym.equation import Equation
import time


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean‑squared error helper (1‑d numpy arrays)."""
    return float(np.mean((y_true - y_pred) ** 2))


ENV_CONFIG: Dict[str, Dict] = {
    "SIRV": dict(
        sim_cls=SIRVOneTimeVaccination,
        prob_cls=Problem_SIRV,
        input_cols=["transmission_rate", "recovery_rate"],
        output_col="vaccinated",
        downsample=True,
        every_n=70,
    ),
    "LAGRANGE": dict(
        sim_cls=Sim_Lagrange,
        prob_cls=Problem_Lagrange,
        input_cols=["distance_b1_b2", "d"],
        output_col=None,
        downsample=False,
        every_n=3,
    ),
    "PLANE": dict(
        sim_cls=Sim_InclinedPlane,
        prob_cls=Problem_InclinedPlane,
        input_cols=["mass", "gravity", "angle"],
        output_col="force",
        downsample=False,
        every_n=1,
    ),
    "DROPFRICTION": dict(
        sim_cls=Sim_DropFriction,
        prob_cls=Problem_DropFriction,
        input_cols=['drop_length', 'adv', 'rec', 'avg_vel', "width"],
        output_col="y",
        downsample=False,
        every_n=1,
    ),
}

SUCCESS_THR: Dict[str, float] = {
    "SIRV":       -0.3,
    "LAGRANGE":   0.9,
    "PLANE": -0.1,
    "DROPFRICTION": 0.1,
}


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


def preprocess_dataframe(args, df):
    if args.simulation == "basketball":
        # normalise per episode
        df["episode"] = (df["time"] < df["time"].shift()).cumsum()
        df = df[df["episode"] < 50]

        def trim(group):
            return group.iloc[:int(len(group) * 0.5)]

        df = df.groupby("episode", group_keys=False).apply(trim).reset_index(drop=True)

        def norm(group, col="ball_y"):
            offset = group[col].iloc[0]
            group[col] = group[col] - offset
            return group

        df = df.groupby("episode", group_keys=False).apply(norm).reset_index(drop=True)
        df["velocity_sin_angle"] = df["velocity"] * np.sin(df["angle"])
        df["g"] = -9.80665

    if args.simulation == "lagrange":
        df["d"] = (
                          df["body_2_mass"] / (df["body_1_mass"] + df["body_2_mass"])
                  ) * df["distance_b1_b2"]

    if args.simulation == "drop_friction":
        loaded_scaler_Y = joblib.load(f'/home/jbrugger/PycharmProjects/Science-gym/environments/drop_friction_models/Teflon-Au-Ethylene Glycol/scaler_Y.pkl')
        df = pd.DataFrame(
            loaded_scaler_Y.inverse_transform(
                df),
            columns=df.columns
        )
    return df


def run_symbolic_regression(args, csv_file, problem) -> List[Dict]:
    """Fit PySR and return a list of result dicts."""
    results_sr = {'overview' : {}}
    if args.simulation == "basketball":
        csv_file =csv_file.with_name(csv_file.stem + "_episodes.csv")
    print(f'Opening {csv_file}')
    df = pd.read_csv(csv_file)
    df = preprocess_dataframe(args, df)

    if args.downsample:
        df = df.iloc[:: args.every_n].reset_index(drop=True)

    if args.simulation == "lagrange":
        targets = [
            ("bod_3_posX", args.input_cols),
            ("bod_3_posY", ["distance_b1_b2"]),
        ]
    else:
        targets = [(args.output_col, args.input_cols)]

    ground_truth = problem.solution()
    for out_col, in_cols in targets:
        print(f'===Regressing for {out_col}===')
        print('Data sample:')
        print(df.head(5))
        y_true = df[out_col].values
        X = df[in_cols].values

        model = PySRRegressor(
            model_selection=args.model_selection,
            niterations=args.niterations,
            binary_operators=args.binary_operators,
            unary_operators=args.unary_operators,
            progress=args.progress,
            random_state=args.seed,
            should_simplify=args.should_simplify,
            deterministic=args.deterministic,
            parallelism=args.parallelism,
            maxsize=args.maxsize,
            complexity_of_constants=args.complexity_of_constants,
            weight_optimize=args.weight_optimize,
        ).fit(X, y_true, variable_names=in_cols)
        if ground_truth is None:
            gt_mse = None

        elif isinstance(ground_truth, list):
            # multiple solutions possible
            min_mse = 1000000
            best_gt = None
            for gt in ground_truth:
                gt_mse = mse(y_true, gt.evaluate(df))
                if gt_mse < min_mse:
                    min_mse = gt_mse
                    best_gt = gt
        else:
            gt_mse = mse(y_true, ground_truth.evaluate(df))

        for i, r in model.equations_.iterrows():
            expr = str(r["sympy_format"])
            eq = Equation(expr)
            evaluation_dict = problem.evaluation(eq, data=df)
            results_sr[i] = dict(
                    target=out_col,
                    equation=expr,
                    complexity=int(eq.complexity()),
                    evaluation=evaluation_dict,
                    gt_mse=gt_mse,
                    pysr_score=r["score"]
                )

            print(f"Equation: {eq}, MSE: {mse}, Equality: {eq == ground_truth}")

        print(f'Expected result: {ground_truth}')

    return results_sr


def main():
    __pre_parser = ConfigGeneral.arguments_parser()
    pre_args, _ = __pre_parser.parse_known_args()

    parser = ConfigGeneral.arguments_parser()
    parser = ConfigRLAgent.arguments_parser(parser)

    if pre_args.equation_discoverer == 'pysr':
        parser = ConfigPySR.arguments_parser(parser)
    if pre_args.equation_discoverer == 'gplearn':
        parser = ConfigGPlearn.arguments_parser(parser)
    if pre_args.simulation == 'basketball':
        parser = ConfigBasketball.arguments_parser(parser)

    args = parser.parse_args()
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

    args.result_dir = (args.root_dir / args.result_dir / args.exp_name / args.rl_agent
                       /args.context/ f"seed_{args.seed}"
                       / f"{datetime.now().strftime('%Y_%b_%d_%H_%M')}")
    print(f"Experiments result will be saved to {args.result_dir}")
    args.result_dir.mkdir(exist_ok=True, parents=True)
    with open(args.result_dir / 'arguments.json', "w") as outfile:
        json.dump(cast_to_serialize(vars(args)), outfile, indent=4)
    print(f"\n=== Training {args.simulation} (context={args.context}) ===")
    start_time = time.time()
    records_dict = {}
    if args.regress_only:
        csv_file = args.root_dir /  args.path_to_regression_table
        if not csv_file.exists():
            raise FileExistsError(f"File {csv_file} does not exist")
        end_time_rl = time.time()
    else:
        print('Training RL agent...')
        csv_file = args.result_dir/ "successful_states.csv"
        agent, problem = train_agent(args, sim, problem)
        end_time_rl = time.time()

        records_dict = record_successful_episodes(
            args,
            agent,
            problem,
            csv_path = csv_file,
            threshold=args.success_thr
        )

    results = run_symbolic_regression(args, csv_file, problem)
    end_time_sr = time.time()

    results['overview']['time_rl'] = end_time_rl - start_time
    results['overview']['time_sr'] = end_time_sr - end_time_rl
    results['overview']['#_record_dict'] = records_dict
    results['overview']['done'] = True

    print(f"Results saved in : { args.result_dir/ 'results_sr.json'}")
    with open(args.result_dir / 'results_sr.json', "w") as outfile:
        json.dump(cast_to_serialize(results), outfile, indent=4)

def cast_to_serialize(element):
    if isinstance(element, dict):
        return {k: cast_to_serialize(v) for k, v in element.items()}
    if isinstance(element, list):
        return [cast_to_serialize(v) for v in element]
    if isinstance(element, tuple):
        return tuple([cast_to_serialize(v) for v in element])
    if isinstance(element, Path):
        return str(element)
    if isinstance(element, np.ndarray):
        return element.tolist()
    return element

if __name__ == "__main__":
    main()
