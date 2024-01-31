import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def derivative_at_point(func, x, h=1e-5):
    """
    Находит производную функции в точке по определению.

    Args:
        func (function): Функция, для которой находится производная.
        x (float): Точка, в которой вычисляется производная.
        h (float): Маленькое приращение x для вычисления производной.

    Returns:
        float: Значение производной в точке.
    """
    try:
        derivative = (func(x + h) - func(x)) / h
    except Exception:
        print(f'Функция не дифференцируема в точке {x}')
        return None
    return derivative

def second_derivative(func, x, h=1e-5):
    def derivative(func, x, h=1e-5):
        return(func(x + h) - func(x)) / h
    (derivative(func, x + h) - derivative(func, x)) / h
    return (derivative(func, x + h) - derivative(func, x)) / h


def create_graph(my_func, x_values):
    import types
    if type(my_func) == types.FunctionType:
        func = my_func
    elif type(my_func) == sp.Expr:
        x = sp.symbols('x')
        func = sp.lambdify(x, my_func, 'numpy')

    # Вычисление значений y с использованием функции
    y_values = []
    for x in x_values:
        y_values.append(func(x))

    # Построение графика
    plt.plot(x_values, y_values, label='')
    plt.title('')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
