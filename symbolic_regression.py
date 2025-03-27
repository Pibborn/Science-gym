import pandas as pd
from pysr import PySRRegressor

# === User inputs ===
csv_path = 'output.csv'  
input_columns = ['susceptible', 'infected', 'recovered', 'transmission_rate', 'recovery_rate']
output_column = 'vaccinated'
# ===================

# Load the data
df = pd.read_csv(csv_path)

print(df.head(5))

# Extract input and output arrays
X = df[input_columns].values
y = df[output_column].values

# Create symbolic regressor
model = PySRRegressor(
    model_selection="best",  
    niterations=40,        
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "sin", "cos", "exp", "log", "sqrt", "abs"
    ],
    extra_sympy_mappings={"sqrt": lambda x: x**0.5},
    progress=True,
)

# Fit the model
model.fit(X, y)

# Print the discovered equations
print(model)

