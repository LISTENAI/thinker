from math import floor, ceil
from typing import Dict, Tuple

from ....xsympy import is_sympy

def get_function(expr: str, fuc: str) -> Tuple[int, int]:
    """Get the start and end indices of a function in the expression string.
    
    Args:
        expr (str): The input expression string.
        fuc (str): The function name to search for.
        
    Returns:
        Tuple[int, int]: A tuple containing the start and end indices of the function.
    """
    if expr.startswith(fuc):
        start = len(fuc) + 1
        end = expr.rfind(')')
        return (start, end)
    return (0, 0)  # Return default values if function not found

def calc_expr(expr: str, dynamic_shape: Dict) -> int:
    """Calculate the value of an expression based on dynamic shape information.
    
    Args:
        expr (str): The expression to evaluate.
        dynamic_shape (Dict): A dictionary containing dynamic shape information.
        
    Returns:
        int: The evaluated integer result of the expression.
    """
    # Define allowed built-in functions to prevent security risks
    allowed_functions = {'floor': floor, 'ceil': ceil, 'Max': max, 'Min': min}
    # Create a safe dictionary for eval
    eval_globals = {'__builtins__': None}  # Disable built-in functions
    eval_globals.update(allowed_functions)
    eval_globals.update(dynamic_shape)
    
    try:
        result = eval(expr, eval_globals)
        return int(result)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expr}': {str(e)}")

__all__ = ['calc_expr']