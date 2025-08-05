from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from gplearn.functions import _Function
from gplearn.genetic import SymbolicRegressor
from sympy import simplify, symbols, sympify
import re

from sciencegym.equation import Equation
from sciencegym.equation_discovery.cast_gp_learn import map_equation_to_syntax_tree


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean‑squared error helper (1‑d numpy arrays)."""
    return float(np.mean((y_true - y_pred) ** 2))


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


def run_symbolic_regression(args, csv_file, problem):
    """Fit PySR and return a list of result dicts."""

    if args.simulation == "basketball":
        csv_file = csv_file.with_name(csv_file.stem + "_episodes.csv")
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
    results_sr = {'overview': {}}
    for out_col, in_cols in targets:
        print(f'===Regressing for {out_col}===')
        print('Data sample:')
        print(df.head(5))

        gt_mse = get_ground_truth_mse_error(
            df,
            ground_truth,
            y_true=df[out_col].values
        )
        if args.equation_discoverer == 'pysr':
            result_dict = run_pysr(args, df, gt_mse, in_cols, out_col, problem)
        elif args.equation_discoverer == 'gplearn':
            result_dict = run_gplearn(args, df, gt_mse, in_cols, out_col, problem)
        else:
            raise NotImplementedError
        results_sr[f"{out_col}, {in_cols}"] = result_dict

    return results_sr


def run_pysr(args, df, gt_mse, in_cols, out_col, problem):
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
    result_dict = {}
    for i, r in model.equations_.iterrows():
        expr = str(r["sympy_format"])
        eq = Equation(expr)
        evaluation_dict = problem.evaluation(eq, data=df)
        result_dict[i] = dict(
            target=out_col,
            equation=expr,
            complexity=int(eq.complexity()),
            evaluation=evaluation_dict,
            gt_mse=gt_mse,
            pysr_score=r["score"]
        )
    return result_dict

def run_gplearn(args, df, gt_mse, in_cols, out_col, problem):
    y_true = df[out_col].values
    X = df[in_cols].values
    op_to_func = {
        '+':'add', '-':'sub', '*':'mul', '/':'div', 'sqrt':'sqrt',
        'log':'log', 'abs':'abs', 'neg':'neg', 'inv':'inv',
        'sin':'sin', 'cos':'cos', 'tan':'tan', '^': 'pow'
    }
    x_to_cols_dict = {}
    for i, col in enumerate(in_cols):
        x_to_cols_dict[f'x_{i}'] = col
    function_set = [op_to_func[op] for op in args.binary_operators]
    [function_set.append(op_to_func[op]) for op in args.unary_operators]
    esp_gp = SymbolicRegressor(population_size=50, generations=args.niterations,
                               max_samples=0.9,
                               parsimony_coefficient=0.01, random_state=args.seed,
                               function_set=function_set,
                               )
    esp_gp.fit(X, y_true)
    top_n = sorted(esp_gp._programs[-1], key=lambda prog: prog.raw_fitness_)[:30]
    result_dict = {}
    seen_gp_prefix ={}
    for i, prog in enumerate(top_n, 1):
        gp_prefix = gp_program_to_prefix(prog.program)
        if not gp_prefix in seen_gp_prefix:
            seen_gp_prefix[gp_prefix] = None
        else:
            continue
        gp_prefix = ' '.join( [x_to_cols_dict[term] if term in x_to_cols_dict else term
                  for term in gp_prefix.split(' ')])
        tree = map_equation_to_syntax_tree(args, gp_prefix, infix=False)
        infix = tree.rearrange_equation_infix_notation()[1]
        try:
            eq = Equation(infix)
            evaluation_dict = problem.evaluation(eq, data=df)
            result_dict[i] = dict(
                target=out_col,
                equation=infix,
                complexity=int(eq.complexity()),
                evaluation=evaluation_dict,
                gt_mse=gt_mse,
                gp_score=prog.fitness_
            )
        except Exception as e:
            print(f"{i}: Failed to simplify: {infix} ({e})")
    return result_dict


def get_ground_truth_mse_error(df, ground_truth, y_true):
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
    return gt_mse


def gp_program_to_prefix(program):
    prefix = ''
    for f in program:
        prefix += ' '
        if isinstance(f,_Function):
            f_str = f.name
            if f_str =='add':
                prefix += '+'
            elif f_str =='sub':
                prefix += '-'
            elif f_str =='pow':
                prefix += '**'
            elif f_str =='mul':
                prefix += '*'
            elif f_str == 'div':
                prefix += '/'
            elif f_str == 'neg':
                prefix += ' - 0 '
            elif f_str == 'inv':
                prefix += ' / 1  '
            elif f_str == 'Abs':
                prefix += ' abs  '
            else:
                prefix += f_str
        elif isinstance(f, int):
            prefix += f'x_{f}'
        elif isinstance(f, float):
            prefix += str(f)
        else:
            raise NotImplementedError('This we should never reach '
                                      f'f is a instance of {f}')
    return prefix