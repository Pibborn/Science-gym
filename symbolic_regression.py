import pandas as pd
from pysr import PySRRegressor
from sciencegym.equation import Equation
from sciencegym.problems.Problem_SIRV import Problem_SIRV
from sciencegym.simulations.Simulation_SIRV import SIRVOneTimeVaccination

# === User inputs ===
csv_path = 'output_inclinedplane.csv'  
#input_columns = ['transmission_rate', 'recovery_rate']
#output_column = 'vaccinated'
#csv_path = 'output_SIRV.csv'
input_columns = ['mass', 'gravity', 'angle']
output_column = 'force'
# ===================

# Load the data
df = pd.read_csv(csv_path)

print(df.head(5))

# Extract input and output arrays
X = df[input_columns].values
y = df[output_column].values
y = y*-1

# Create symbolic regressor
model = PySRRegressor(
    model_selection="best",  
    niterations=1,        
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[        
    ],
    extra_sympy_mappings={"sqrt": lambda x: x**0.5},
    progress=True,
)

model.fit(X, y, variable_names=input_columns)

env = SIRVOneTimeVaccination()
problem = Problem_SIRV(env)

for _, row in model.equations_.iterrows():
    expr = row['sympy_format']
    eq = Equation(expr)
    score = problem.evaluate(eq, data=df)
    print(f"Equation: {eq}, Score: {score}, Equality: {eq == problem.solution()}")