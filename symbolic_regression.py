import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sciencegym.equation import Equation
from sciencegym.problems.Problem_SIRV import Problem_SIRV
from sciencegym.simulations.Simulation_SIRV import SIRVOneTimeVaccination
import matplotlib.pyplot as plt

downsample = False
# === User inputs ===
environments = ['SIRV', 'INCLINEDPLANE','BASKETBALL' ]
current_env = environments[2]
if current_env == 'SIRV':
    input_columns = ['transmission_rate', 'recovery_rate']
    output_column = 'vaccinated'
    csv_path = 'output_SIRV.csv'
elif current_env == 'INCLINEDPLANE':
    csv_path = 'output_inclinedplane.csv' 
    input_columns = ['mass', 'gravity', 'angle']
    output_column = 'force'
elif current_env == 'BASKETBALL':
    csv_path = 'output_episodes.csv' 
    input_columns = ['velocity', 'angle', 'time', 'g']
    output_column = 'ball_y'

    downsample = True
    every_n_row = 10
else:
    raise ValueError('Specified environment not found!')
# ===================

# Load the data
df = pd.read_csv(csv_path)

# Downsample the number of rows
if downsample:
    df = df.iloc[::every_n_row].reset_index(drop=True)

print(f"Length of downsample: {len(df)}")

# Extract input and output arrays
X = df[input_columns].values
y = df[output_column].values
#y = y*-1

# Create symbolic regressor
model = PySRRegressor(
    model_selection="best",  
    niterations=50,        
    binary_operators=["*",'-'],
    unary_operators=['sin'],
    extra_sympy_mappings={"sqrt": lambda x: x**0.5},
    progress=True,
)

model.fit(X, y, variable_names=input_columns)
print(model.get_best())
print(model.equations_)



env = SIRVOneTimeVaccination()
problem = Problem_SIRV(env)

for _, row in model.equations_.iterrows():
    expr = row['sympy_format']
    eq = Equation(expr)
    score = problem.evaluate(eq, data=df)
    print(f"Equation: {eq}, Score: {score}, Equality: {eq == problem.solution()}")