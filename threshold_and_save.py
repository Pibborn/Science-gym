import os
import csv
from pathlib import Path
from itertools import product
from typing import Dict, List

import numpy as np
import pandas as pd

from sciencegym.agents.StableBaselinesAgents.SACAgent import SACAgent
from sciencegym.simulations.Simulation_Basketball import Sim_Basketball
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

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean‑squared error helper (1‑d numpy arrays)."""
    return float(np.mean((y_true - y_pred) ** 2))

ENV_CONFIG: Dict[str, Dict] = {
    "BASKETBALL": dict(
        sim_cls=Sim_Basketball,
        prob_cls=Problem_Basketball,
        input_cols=["velocity_sin_angle", "time", "g"],
        output_col="ball_y",
        downsample=False,
        every_n=10,
    ),
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
        downsample=True,
        every_n=10,
    ),
}

SUCCESS_THR: Dict[str, float] = {
    "BASKETBALL": 80,
    "SIRV":       90,
    "LAGRANGE":   0.7,
    "PLANE": 0.8
}

TIMESTEPS = 200_000
TEST_EPISODES = 10_000  
RESULTS_DIR = Path("results")

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


def train_agent(sim, problem_cls):
    """Return a trained SACAgent on `sim`."""
    problem = problem_cls(sim)
    vec_env = DummyVecEnv([lambda: problem])
    in_dim, out_dim = get_env_dims(sim)
    agent = SACAgent(in_dim, out_dim, lr=1e-4, policy="MlpPolicy")
    model = agent.create_model(vec_env, verbose=0, use_sde=False)
    model.learn(TIMESTEPS, log_interval=1000)
    agent.agent = model
    return agent, problem


def record_successful_episodes(agent, problem, csv_path, threshold):
    """Roll out evaluation episodes, store those above reward threshold."""
    vec_env = DummyVecEnv([lambda: problem])
    successes = []

    for _ in range(TEST_EPISODES):
        obs = vec_env.reset()
        done, reward_sum = False, 0.0
        while not done:
            action, _ = agent.agent.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            reward_sum += reward
            if done:
                terminal_obs = info[0]["terminal_observation"]
                episode = info[0].get("record_episode", np.array([np.nan]))
                if reward_sum >= threshold:
                    successes.append((terminal_obs, episode))
                break

    if not successes:
        print("No successful episodes found ‑ nothing to record.")
        return 0 

    # flatten and write
    states, episodes = zip(*successes)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(problem.variables)
        writer.writerows([s.flatten() for s in states])

    with open(csv_path.with_name(csv_path.stem + "_episodes.csv"), "w",
              newline="") as f:
        writer = csv.writer(f)
        writer.writerow(problem.variables + ["time"])
        print(e)
        writer.writerows([e.flatten() for e in episodes])

    print(f"Saved {len(states)} successful trajectories to {csv_path}")
    return len(states)


def preprocess_dataframe(df, env_key):
    """Match the logic from original symbolic_regression.py."""
    if env_key == "BASKETBALL":
        # normalise per episode
        df["episode"] = (df["time"] < df["time"].shift()).cumsum()
        df = df[df["episode"] < 3]

        def trim(group):
            return group.iloc[: int(len(group) * 0.5)]

        df = df.groupby("episode", group_keys=False).apply(trim).reset_index(drop=True)

        def norm(group, col="ball_y"):
            offset = group[col].iloc[0]
            group[col] = group[col] - offset
            return group

        df = df.groupby("episode", group_keys=False).apply(norm).reset_index(drop=True)
        df["velocity_sin_angle"] = df["velocity"] * np.sin(df["angle"])
        df["g"] = 9.80665

    if env_key == "LAGRANGE":
        df["d"] = (
            df["body_2_mass"] / (df["body_1_mass"] + df["body_2_mass"])
        ) * df["distance_b1_b2"]

    return df


def run_symbolic_regression(csv_path, cfg, env_key, problem) -> List[Dict]:
    """Fit PySR and return a list of result dicts."""
    rows = []
    df = pd.read_csv(csv_path)
    df = preprocess_dataframe(df, env_key)

    if cfg["downsample"]:
        df = df.iloc[:: cfg["every_n"]].reset_index(drop=True)

    if env_key == "LAGRANGE":
        targets = [
            ("bod_3_posX", cfg["input_cols"]),
            ("bod_3_posY", ["distance_b1_b2"]),
        ]
    else:
        targets = [(cfg["output_col"], cfg["input_cols"])]

    ground_truth = problem.solution()
    for out_col, in_cols in targets:
        y_true = df[out_col].values
        X = df[in_cols].values

        model = PySRRegressor(
            model_selection="best",
            niterations=40,
            binary_operators=["*", "-", "+", "/"],
            unary_operators=['sqrt', 'sin', 'cos'],
            progress=False,
        ).fit(X, y_true, variable_names=in_cols)
        
        if isinstance(ground_truth, list):
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

        for _, r in model.equations_.iterrows():
            expr = str(r["sympy_format"])
            eq = Equation(expr)
            y_pred = eq.evaluate(df)
            rows.append(
                dict(
                    environment=env_key,
                    context=problem.simulation.context,
                    target=out_col,
                    equation=expr,
                    complexity=int(eq.complexity()),
                    mse=mse(y_true, y_pred),
                    gt_mse=gt_mse,
                )
            )

    return rows


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    for env_key, ctx in product(ENV_CONFIG.keys(), (0, 1, 2)):
        cfg = ENV_CONFIG[env_key]
        thr = SUCCESS_THR[env_key]
        print(f"\n=== Training {env_key} (context={ctx}) ===")
        
        if 'basket' in str(cfg["sim_cls"]).lower():
            print('Basketball: rendering off')
            kwargs = {'rendering': False}
        else:
            kwargs = {}
        sim = cfg["sim_cls"](context=ctx, **kwargs)
        agent, problem = train_agent(sim, cfg["prob_cls"])

        out_dir = RESULTS_DIR / f"{env_key}_ctx{ctx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_file = out_dir / "successful_states.csv"

        n_saved = record_successful_episodes(agent, problem, csv_file, threshold=thr)
        if n_saved == 0:
            continue
        run_symbolic_regression(csv_file, cfg, env_key, problem)


if __name__ == "__main__":
    main()

