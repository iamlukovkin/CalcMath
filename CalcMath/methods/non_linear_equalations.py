import math
from .. import utils

def division(func, a, b, epsilon):
    '''
    Вычисляет корень уравнения методом деления отрезка пополам.

    Аргументы:
    - func: Функция, уравнение которой нужно решить.
    - a: Левый конец отрезка.
    - b: Правый конец отрезка.
    - epsilon: Точность вычислений.

    Возвращает:
    Результат вычисления корня уравнения.
    '''
    values = [a, b]
    x = (values[1] + values[0]) / 2
    if func(x) == 0:
        return x
    elif func(values[0]) == 0:
        return values[0]
    elif func(values[1]) == 0:
        return values[1]

    while not math.fabs(values[1] - values[0]) < 2 * epsilon:
        x = (values[1] + values[0]) / 2
        if func(x) == 0:
            return x
        if func(values[0]) * func(x) < 0:
            values = [values[0], x]
        elif func(values[1]) * func(x) < 0:
            values = [x, values[1]]

    return (values[1] + values[0]) / 2

def chords_method(func, a, b, epsilon):
    '''
    Вычисляет корень уравнения методом хорд.

    Аргументы:
    - func: Функция, уравнение которой нужно решить.
    - a: Левый конец отрезка.
    - b: Правый конец отрезка.
    - epsilon: Точность вычислений.

    Возвращает:
    Результат вычисления корня уравнения.
    '''
    from .. import utils

    def formula_2(func, x, value):
        result = x - (value - x) / (func(value) - func(x)) * func(x)
        return result

    def formula_3(func, x, value):
        result = x - (x - value) / (func(x) - func(value)) * func(x)
        return result

    if func(b) * utils.second_derivative(func=func, x=b) > 0:
        calc_func = formula_2
        x = b
        value = b
    elif func(a) * utils.second_derivative(func=func, x=a) > 0:
        calc_func = formula_3
        x = a
        value = a
    else:
        print('Невозможно использовать данный метод при предложенных данных!')
        return math.nan

    x_next = a - (b - a) / (func(b) - func(a)) * func(a)

    while not math.fabs(x_next - x) <= epsilon:
        x = x_next
        x_next = calc_func(
            func=func,
            x=x_next,
            value=value
        )

    return x_next

def newton_method(func, a, b, epsilon):
    '''
    Вычисляет корень уравнения методом Ньютона.

    Аргументы:
    - func: Функция, уравнение которой нужно решить.
    - a: Левый конец отрезка.
    - b: Правый конец отрезка.
    - epsilon: Точность вычислений.

    Возвращает:
    Результат вычисления корня уравнения.
    '''

    def formula_6(x):
        result = x - func(x) / utils.derivative_at_point(func=func, x=x)
        return result

    if func(a) * utils.second_derivative(func=func, x=a) > 0:
        x = a
    elif func(b)  * utils.second_derivative(func=func, x=b) > 0:
        x = b
    else:
        print('Функция расходится! Метод не может дать точного результата!')
        return None

    try:
        x_next = formula_6(x)
        while not math.fabs(x_next - x) <= epsilon:
            x = x_next
            x_next = formula_6(x)
    except Exception:
        print('Ошибка в вычислении производной!')
        return None

    return x_next

def iteration_method(func, a, b, epsilon, phi_func=None):
    '''
    Вычисляет корень уравнения методом простых итераций.

    Аргументы:
    - func: Функция, уравнение которой нужно решить.
    - a: Левый конец отрезка.
    - b: Правый конец отрезка.
    - epsilon: Точность вычислений.
    - phi_func: Функция для метода простых итераций. По умолчанию используется автоматически вычисленная функция.

    Возвращает:
    Результат вычисления корня уравнения.
    '''
    def phi(x):
        h = 1e-6
        derivative = (func(x + h) - func(x)) / h
        return x - func(x) / derivative

    if phi_func is not None:
        phi = phi_func

    def is_convergent():
        x = a
        while x <= b:
            derivative = utils.derivative_at_point(phi, x)
            if derivative is None or abs(derivative) >= 1:
                return False
            x += epsilon
        return True

    def requires(x):
        if utils.derivative_at_point(phi, x) is not None:
            derivative_value = abs(utils.derivative_at_point(phi, x))
            return 0 <= derivative_value <= 1
        else:
            return False

    def q_value(x_values):
        return abs(x_values[0] - x_values[1]) / abs(x_values[1] - abs(x_values[2]))

    def exit_require(x_values, q):
        return abs(x_values[0] - x_values[1]) < abs((1 - q) / q) * epsilon

    if not is_convergent():
        print("На данном интервале невозможно провести итерационный метод.")
        return None

    x = a
    while x <= b:
        try:
            if requires(x):
                x_values = [x]
            for _ in range(3):
                if requires(x_values[0]):
                    x_values = [phi(x_values[0]), *x_values]
            while not exit_require(x_values, q_value(x_values)):
                x_values = [phi(x_values[0]), *x_values]
            return x_values[0]
        except Exception:
            print(f'При {x_values[0]} итерационный процесс расходится!')
            return None
        x += epsilon
    return None
