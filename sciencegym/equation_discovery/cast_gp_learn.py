import re

import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import traceback
from sciencegym.SyntaxTree.src.syntax_tree.syntax_tree import SyntaxTree

def map_equation_to_syntax_tree(args, equation, infix=True, catch_exceptions=False ):
    tree = SyntaxTree(grammar=None, args=args)
    if catch_exceptions:
        try:
            if infix:
                best_equation_prefix = infix_to_prefix(equation, args)
            else:
                best_equation_prefix = equation
            tree.prefix_to_syntax_tree(best_equation_prefix.split())
            return tree
        except (SyntaxError, RuntimeError) as e:
            print(f'Can not transform {equation} to syntax tree: {e}')
            print(traceback.format_exc())
            return None
    else:
        if infix:
            best_equation_prefix = infix_to_prefix(equation, args)
        else:
            best_equation_prefix = equation
        tree.prefix_to_syntax_tree(best_equation_prefix.split())
        return tree


def infix_to_prefix(best_equation, args):
    tree = SyntaxTree(grammar=None, args=args)
    obj = InfixToPrefix(possible_operator_2dim='/ ^ * - + '.split(),
                        possible_operator_1dim='ln sin cos root sqrt square cube exp log inv abs tan'.split(),
                        possible_operands='')
    best_equation = best_equation.replace('Abs', ' abs ')
    best_equation = best_equation.replace('inv', ' 1 / ')
    best_equation = best_equation.replace('^', ' ^ ')
    best_equation = best_equation.replace('**', ' ^ ')
    best_equation = best_equation.replace('*', ' * ')
    best_equation = best_equation.replace('+', ' + ')

    best_equation = best_equation.replace('/', ' / ')
    best_equation = best_equation.replace('(', ' ( ')
    best_equation = best_equation.replace(')', ' ) ')
    best_equation = replace_one_argument_with_two(
        best_equation,
        'square',
        '^ 2'
    )
    best_equation = replace_one_argument_with_two(
        best_equation,
        'cube',
        '^ 3'
    )
    best_equation = replace_one_argument_with_two(
        best_equation,
        'root',
        '^ 0.5'
    )
    best_equation = replace_minus_with_two_operator(best_equation)
    best_equation = re.sub(r'-((?!<=e)(?!(\d)))', ' - ', best_equation)
    for i in range(10):
        best_equation = best_equation.replace(f'x{i}', f' x_{i} ')
    try:
        prefix = obj.infixToPrefix(best_equation.split())
    except Exception as e:
        print(f"{best_equation.split()} could not be passed")
        raise e
    prefix = prefix.replace('^', '**')
    return prefix


def replace_minus_with_two_operator(equation):
    equation_array = equation.split()
    two_operator_equation = []
    for i, token in enumerate(equation_array):
        if token.startswith('-') and len(token) > 1:
            two_operator_equation.extend([' ( ', ' 0 ', ' - ', ' 1 ', ' ) ', ' * '  , token[1:] ])
        elif token == '-' and i ==0:  # -()
            two_operator_equation.extend([' ( ', ' 0 ', ' - ', ' 1 ', ' ) * '])
        elif token == '-' and equation_array[i-1] == '(': # (-
            two_operator_equation.extend([' ( ', ' 0 ', ' - ', ' 1 ', ' ) * '])
        else:
            two_operator_equation.append(token)
    return ' '.join(two_operator_equation)


def replace_one_argument_with_two(string, one_arg, two_arg):
    if one_arg in string:
        str_list = string.split(' ')
        first = -1
        last = np.inf
        stack = []
        for i, s in enumerate(str_list):
            if s == one_arg and first == -1:
                first = i
            elif s == '(' and first > -1:
                stack.append('(')
            elif s == ')' and last == np.inf:
                stack.pop()
                if len(stack) == 0:
                    last = i
            else:
                continue
        operator, operand = two_arg.split(' ')
        str_list.insert(last + 1, operand)
        str_list.insert(last + 1, operator)
        del str_list[first]
        string = ' '.join(str_list)
        string = replace_one_argument_with_two(string, one_arg, two_arg)
        return string
    else:
        return string
