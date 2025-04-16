import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sciencegym.equation import Equation

from sciencegym.problems.Problem_SIRV import Problem_SIRV
from sciencegym.problems.Problem_Basketball import Problem_Basketball
from sciencegym.problems.Problem_InclinedPlane import Problem_InclinedPlane
from sciencegym.problems.Problem_Brachistochrone import Problem_Brachistochrone
from sciencegym.problems.Problem_Lagrange import Problem_Lagrange

from sciencegym.simulations.Simulation_SIRV import SIRVOneTimeVaccination
from sciencegym.simulations.Simulaton_Basketball import Sim_Basketball
from sciencegym.simulations.Simulation_InclinedPlane import Sim_InclinedPlane
from sciencegym.simulations.Simulation_Brachistochrone import Sim_Brachistochrone
from sciencegym.simulations.Simulation_Lagrange import Sim_Lagrange

import matplotlib.pyplot as plt

downsample = False
# === User inputs ===
environments = ['SIRV', 'INCLINEDPLANE','BASKETBALL', 'LAGRANGE_L4_X', 'LAGRANGE_L4_Y' ]
current_env = environments[1]
if current_env == 'SIRV':
    input_columns = ['transmission_rate', 'recovery_rate']
    output_column = 'vaccinated'
    csv_path = 'output_SIRV.csv'
    env = SIRVOneTimeVaccination()
    problem = Problem_SIRV(env)
elif current_env == 'INCLINEDPLANE':
    csv_path = 'output_inclinedplane.csv' 
    input_columns = ['mass', 'gravity', 'angle']
    output_column = 'force'
    env = Sim_InclinedPlane()
    problem = Problem_InclinedPlane(env)
elif current_env == 'BASKETBALL':
    csv_path = 'output_episodes.csv' 
    input_columns = ['velocity', 'angle', 'time', 'g']
    output_column = 'ball_y'

    downsample = True
    every_n_row = 10

    env = Sim_Basketball()
    problem = Problem_Basketball(env)
elif current_env == 'LAGRANGE_L4_Y':
    csv_path = 'output_Lagrange.csv' 
    input_columns = ['distance_b1_b2']#['body_1_mass', 'body_2_mass', 'distance_b1_b2', 'bod_3_posX', 'bod_3_posY']
    output_column = 'bod_3_posY'

    env = Sim_Lagrange()
    problem = Problem_Lagrange(env)
elif current_env == 'LAGRANGE_L4_X':
    csv_path = 'output_Lagrange.csv' 
    input_columns = ['distance_b1_b2', 'd']#['body_1_mass', 'body_2_mass', 'distance_b1_b2', 'bod_3_posX', 'bod_3_posY']
    output_column = 'bod_3_posX'
    downsample = True
    every_n_row = 3

    env = Sim_Lagrange()
    problem = Problem_Lagrange(env)
else:
    raise ValueError('Specified environment not found!')
# ===================

csv_path = 'output.csv'

# Load the data
df = pd.read_csv(csv_path)

if current_env.startswith('LAGRANGE'):
    df['d'] = (df['body_2_mass'] /(df['body_1_mass'] + df['body_2_mass']))  * df['distance_b1_b2']

# Downsample the number of rows
if downsample:
    df = df.iloc[::every_n_row].reset_index(drop=True)

print(f"Length of df: {len(df)}")

# Extract input and output arrays
X = df[input_columns].values
y = df[output_column].values
y = y*-1

# Create symbolic regressor
model = PySRRegressor(
    model_selection="best",  
    niterations=40,        
    binary_operators=["*"],
    unary_operators=['sin'],
    extra_sympy_mappings={"sqrt": lambda x: x**0.5},
    progress=True,
)

model.fit(X, y, variable_names=input_columns)
print(model.get_best())
print(model.equations_)


for _, row in model.equations_.iterrows():
    expr = row['sympy_format']
    eq = Equation(expr)
    score = problem.evaluation(eq, data=df)
    if type(score) is list:
        for i, s in enumerate(score):
            print(f"Equation: {eq}, Score: {score}, Equality: {eq == problem.solution()}")
    else:
        print(f"Equation: {eq}, Score: {score}, Equality: {eq == problem.solution()}")