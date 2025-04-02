import sympy as sp
import numpy as np
import pandas as pd

class Equation:
    def __init__(self, expr_str):
        self.expr_str = expr_str
        self.expr = sp.sympify(expr_str)
        self.variables = sorted(str(s) for s in self.expr.free_symbols)
        self.symbols = {v: sp.Symbol(v) for v in self.variables}
        self._f = sp.lambdify([list(self.symbols.values())], self.expr, modules='numpy')

    def evaluate(self, df: pd.DataFrame):
        """
        Evaluates the equation on a pandas DataFrame.
        Only uses the variables actually present in the equation.
        """
        try:
            values = [df[v].values for v in self.variables]
        except KeyError as e:
            raise ValueError(f"Missing variable in data: {e}")
        return self._f(values)

    def complexity(self):
        return sp.count_ops(self.expr)

    def __eq__(self, other):
        return sp.simplify(self.expr - other.expr) == 0

    def __str__(self):
        return str(self.expr)

    def to_sympy(self):
        return self.expr
