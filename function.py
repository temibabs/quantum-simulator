from typing import Union, Any

from numpy import ndarray, array, fft, random
from sympy import fourier_transform, exp, lambdify, latex, abc
from sympy.core import basic
from sympy.parsing.sympy_parser import parse_expr


class VariableNotFoundError(Exception):
    """Variable not found error.
    """

    def __str__(self):
        """Print this exception.
        """
        return "Variable not found"


def dirac(x: ndarray) -> ndarray:
    """
    Approximation of the Dirac delta function.
    :param x:
    :return:
    """

    try:
        dx = (x.max() - x.min()) / len(x)
        return array([1e10 if (-dx / 2 < xi < dx / 2) else 0. for xi in x])
    except TypeError:
        return array([1e10 if (0.01 > x > -0.01) else 0.])


def fourier(x: ndarray) -> ndarray:
    try:
        return fft

    except Exception as e:
        print(e.__cause__)
        return array(1)


def rect(x: ndarray) -> Union[ndarray, float]:
    try:
        return array([1.0 if (-0.5 < x_i < 0.5) else 0. for x_i in x])
    except TypeError:
        return 1.0 if (-0.5 < x < 0.5) else 0.


def noise(x: ndarray) -> Union[ndarray, float]:
    if isinstance(x, ndarray):
        return array([2.0 * random.rand() + 1.0 for _ in range(len(x))])
    else:
        return 2.0 * random.rand() - 1.0


def multiplies_var(main_var: basic.Basic, arb_var: basic.Basic,
                   expr: basic.Basic) -> bool:
    arg_list = []
    for arg1 in expr.args:
        if arg1.has(main_var):
            arg_list.append(arg1)
            for arg2 in expr.args:
                if ((arg2 is arb_var or (arg2.is_Pow and arg2.has(arb_var)))
                        and expr.has(arg1 * arg2)):
                    return True
    return any([multiplies_var(main_var, arb_var, arg)
                for arg in arg_list if
                (arg is not main_var)])


class Function:

    def __init__(self, function_name: str,
                 param: Union[basic.Basic, str]) -> None:
        """
        The initializer. The parameter must be a
        string representation of a function, and it needs to
        be at least a function of x.
        """
        # Dictionary of modules and user defined functions.
        # Used for lambdify from sympy to parse input.
        if isinstance(param, str):
            param = parse_expr(param)
        if function_name == "x":
            function_name = "1.0*x"
        self._symbolic_func = parse_expr(function_name)
        symbol_set = self._symbolic_func.free_symbols
        if abc.k in symbol_set:
            k_param = parse_expr("k_param")
            self._symbolic_func = self._symbolic_func.subs(abc.k, k_param)
            symbol_set = self._symbolic_func.free_symbols
        symbol_list = list(symbol_set)
        if param not in symbol_list:
            raise VariableNotFoundError
        self.latex_repr = latex(self._symbolic_func)
        symbol_list.remove(param)
        self.parameters = symbol_list
        var_list = [param]
        var_list.extend(symbol_list)
        self.symbols = var_list
        self._lambda_func = lambdify(
            self.symbols, self._symbolic_func, modules=self.module_list)

    module_list = ['numpy', {'rect': rect, 'noise': noise}]

    @staticmethod
    def add_function(function_name, function) -> None:
        Function.module_list[1][function_name] = function

    def __call__(self, x: Union[ndarray, float], *args: float) -> ndarray:
        if args == ():
            kwargs = self.get_default_values()
            args = [kwargs[s] for s in kwargs]
        return self._lambda_func(x, *args)

    def get_default_values(self):
        return {s: float(multiplies_var(self.symbols[s], s, self._symbolic_func))
                for s in self.parameters}

    def get_enumerated_default_values(self):
        return {i: [s, float(multiplies_var(self.symbols[s], s, self._symbolic_func))]
                for i, s in enumerate(self.parameters)}

    def get_tupled_default_values(self) -> tuple:
        defaults = self.get_enumerated_default_values()
        return tuple([defaults[i][1] for i in range(len(self.parameters))])

    def multiply_latex_string(self, var: str) -> str:
        var = parse_expr(var)
        expr = var * self._symbolic_func
        return latex(expr)


if __name__ == '__main__':
    print(fourier(array([0.25, 3.5, 2.3, 0.4])))
