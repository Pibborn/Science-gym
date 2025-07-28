import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sciencegym.equation import Equation


from sciencegym.problems.Problem_SIRV import Problem_SIRV
from sciencegym.problems.Problem_Basketball import Problem_Basketball
from sciencegym.problems.Problem_InclinedPlane import Problem_InclinedPlane
from sciencegym.problems.Problem_Brachistochrone import Problem_Brachistochrone
from sciencegym.problems.Problem_Lagrange import Problem_Lagrange
from sciencegym.problems.Problem_DropFriction import Problem_DropFriction

from sciencegym.simulations.Simulation_SIRV import SIRVOneTimeVaccination
from sciencegym.simulations.Simulaton_Basketball import Sim_Basketball
from sciencegym.simulations.Simulation_InclinedPlane import Sim_InclinedPlane
from sciencegym.simulations.Simulation_Brachistochrone import Sim_Brachistochrone
from sciencegym.simulations.Simulation_Lagrange import Sim_Lagrange
from sciencegym.simulations.Simulation_DropFriction import Sim_DropFriction

import matplotlib.pyplot as plt

downsample = False
# === User inputs ===
environments = ['SIRV', 'INCLINEDPLANE','BASKETBALL', 'LAGRANGE_L4_X', 'LAGRANGE_L4_Y', 'DROPFRICTION' ]
current_env = environments[3]
if current_env == 'SIRV':
    input_columns = ['transmission_rate', 'recovery_rate']
    output_column = 'vaccinated'
    csv_path = 'output_SIRV.csv'

    downsample = True
    every_n_row = 70

    env = SIRVOneTimeVaccination()
    problem = Problem_SIRV(env)
elif current_env == 'INCLINEDPLANE':
    csv_path = 'output_inclinedplane.csv' 
    input_columns = ['mass', 'gravity', 'angle']
    output_column = 'force'

    downsample = True
    every_n_row = 10

    env = Sim_InclinedPlane()
    problem = Problem_InclinedPlane(env)
elif current_env == 'BASKETBALL':
    csv_path = 'output_episodes.csv' 
    input_columns = ['velocity', 'angle', 'time', 'g']
    output_column = 'ball_y'

    downsample = False
    every_n_row = 10

    env = Sim_Basketball()
    problem = Problem_Basketball(env)
elif current_env == 'LAGRANGE_L4_Y':

    csv_path = 'output_Lagrange.csv'
    input_columns = ['distance_b1_b2']#['body_1_mass', 'body_2_mass', 'distance_b1_b2', 'bod_3_posX', 'bod_3_posY']
    output_column = 'bod_3_posY'
    downsample = True
    every_n_row = 10
    env = Sim_Lagrange()
    problem = Problem_Lagrange(env)
elif current_env == 'LAGRANGE_L4_X':
    csv_path = 'output_Lagrange.csv'
    input_columns = ['distance_b1_b2', 'd']#['body_1_mass', 'body_2_mass', 'distance_b1_b2', 'bod_3_posX', 'bod_3_posY']
    output_column = 'bod_3_posX'
    downsample = False
    every_n_row = 3

    env = Sim_Lagrange()
    problem = Problem_Lagrange(env)
elif current_env == 'DROPFRICTION':
    csv_path = 'output_DROPFRICTION.csv'
    input_columns = ['drop_length', 'adv', 'rec', 'avg_vel', 'width']
    output_column = 'y'
    downsample = False
    every_n_row = 1

    env = Sim_DropFriction()
    problem = Problem_DropFriction()
else:
    raise ValueError('Specified environment not found!')
# ===================

#csv_path = 'output.csv'

# Load the data
df = pd.read_csv(csv_path)
df['g'] = 9.80665


if current_env.startswith('BASKETBALL'):
    def trim(group):
            cutoff = int(len(group) * 0.5)
            return group.iloc[:cutoff]

    df['episode'] = (df['time'] < df['time'].shift()).cumsum()
    df = df[df['episode'] < 3]
    df = df.groupby('episode', group_keys=False).apply(trim)
    df = df.reset_index(drop=True)

    def normalize_ball_y(df, col='ball_y'):
        df = df.copy()
        offset = df[col].iloc[0]
        df[col] = df[col] - offset
        return df

    def normalize_ball_y_per_episode(df, time_col='time', col='ball_y'):
        df = df.copy()
        df['episode'] = (df[time_col] < df[time_col].shift()).cumsum()

        def normalize(group):
            offset = group[col].iloc[0]
            group[col] = group[col] - offset
            return group

        normalized_df = df.groupby('episode', group_keys=False).apply(normalize)
        return normalized_df.reset_index(drop=True)

    df = normalize_ball_y_per_episode(df)

    df['velocity_sin_angle'] = df['velocity'] * np.sin(df['angle'])



if current_env.startswith('LAGRANGE'):
    df['d'] = (df['body_2_mass'] /(df['body_1_mass'] + df['body_2_mass']))  * df['distance_b1_b2']

# Downsample the number of rows
if downsample:
    df = df.iloc[::every_n_row].reset_index(drop=True)

print(f"Length of df: {len(df)}")

# Extract input and output arrays
X = df[input_columns].values
y = df[output_column].values
#y = y*-1

# Create symbolic regressor
model = PySRRegressor(
    model_selection="best",  
    niterations=40,
    binary_operators=["*", "-"],
    unary_operators=[],#"sin", "square"],
    extra_sympy_mappings={"sqrt": lambda x: x**0.5},
    progress=True,
)

model.fit(X, y, variable_names=input_columns)
print(model.get_best())
print(model.equations_)


for _, row in model.equations_.iterrows():
    expr = row['sympy_format']
    eq = Equation(expr)
    solutions = problem.solution()
    score = problem.evaluation(eq, data=df)
    if type(score) is list:
        for i, s in enumerate(score):
            print(f"Equation: {eq}, Solution{solutions[i]}, Score: {score}, Equality: {eq == solutions[i]}")
    else:
        print(f"Equation: {eq}, Score: {score}, Equality: {eq == problem.solution()}")