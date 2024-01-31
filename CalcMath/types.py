import time
import math

import numpy as np

from .methods import integration as IM
from .methods import non_linear_equalations as NLE
from .methods import linear_equations as LE
from .methods import interpolation_functions as IntFunc


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        print(f'Метод: {func.__name__}.')
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Время выполнения: {execution_time:.6f} секунд.\nРезультат:\n{result}")
        return result
    return wrapper


def interpolation_decorator(func):
    def wrapper(*args, **kwargs):
        print(f'\nМетод: {func.__name__}.')
        start_time = time.time()
        result = func(*args, **kwargs)
        import types
        if type(result) == types.FunctionType:
            string_result = result.__name__
        else:
            string_result = result
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Время выполнения: {execution_time:.6f} секунд.\nПолученное уравнение:\n{string_result}")
        return result
    return wrapper


class InterpolationFunctions:
    def __init__(
        self, 
        x_values: list,
        y_values: list,
    ):
        self.x_list = x_values
        self.y_list = y_values
    
    @interpolation_decorator
    def lagrange_method(self):
        return IntFunc.lagrange_interpolation(x_data=self.x_list, y_data=self.y_list)
    
    @interpolation_decorator
    def aitken_method(self):
        return IntFunc.aitken_interpolation(x_data=self.x_list, y_data=self.y_list)
    
    @interpolation_decorator
    def spline_method(self):
        return IntFunc.cubic_spline(x_data=self.x_list, y_data=self.y_list)
    


class Integration:
    """
    Класс для выполнения численного интегрирования методами прямоугольников, трапеций и Симпсона.

    Аргументы:
    - func: функция, которую нужно проинтегрировать
    - a: нижний предел интегрирования
    - b: верхний предел интегрирования
    - epsilon: погрешность вычислений

    Методы:
    - calc_divided_values(self, func): Вычисляет интеграл функции `func` с заданной погрешностью `epsilon`.
    - left_rectangular(self): Метод левых прямоугольников для интегрирования.
    - central_rectangular(self): Метод центральных прямоугольников для интегрирования.
    - right_rectangular(self): Метод правых прямоугольников для интегрирования.
    - trapezoid_rule(self): Метод трапеций для интегрирования.
    - simpson_rule(self): Метод Симпсона для интегрирования.
    - newton_rule(self): Метод Ньютона для интегрирования.

    Примечание: Для вызова методов интегрирования, передайте функцию интегрирования (например, self.left_rectangular()).

    """

    def __init__(self, func, a: float, b: float, epsilon: float):  
        self.func = func
        self.a = a
        self.b = b
        self.epsilon = epsilon
    
    def calc_divided_values(self, func):
        """
        Вычисляет интеграл функции с заданной погрешностью.

        Аргументы:
        - func: Функция для интегрирования.

        Возвращает:
        Результат численного интегрирования и количество разбиений.

        """
        if not 0 < self.epsilon < 1:
            print('Выбрана неверная погрешность!')
            return None, math.nan
        self.n = 1
        if func == self.left_rectangular or func == self.right_rectangular:
            P = 1
        elif func == self.central_rectangular or func == self.trapezoid_rule:
            P = 2
        elif func == self.simpson_rule or func == self.newton_rule:
            P = 4
        else: 
            print('Выбрана неверная функция!')
            return None, math.nan
        self.n = 1
        I1 = func()
        self.n *= 2
        I2 = func()
        while not 1 / (2 ** P - 1) * math.fabs(I1 - I2) < self.epsilon:
            I1 = func()
            self.n *= 2
            I2 = func()
        return I2, self.n
    
    @timing_decorator
    def left_rectangular(self):
        """
        Метод левых прямоугольников для интегрирования.

        Возвращает:
        Результат численного интегрирования.

        """
        return IM.left_rectangular(func=self.func, a=self.a, b=self.b, n=self.n)
    
    @timing_decorator
    def central_rectangular(self):
        """
        Метод центральных прямоугольников для интегрирования.

        Возвращает:
        Результат численного интегрирования.

        """
        return IM.central_rectangular(func=self.func, a=self.a, b=self.b, n=self.n)
    
    @timing_decorator
    def right_rectangular(self):
        """
        Метод правых прямоугольников для интегрирования.

        Возвращает:
        Результат численного интегрирования.

        """
        return IM.right_rectangular(func=self.func, a=self.a, b=self.b, n=self.n)
    
    @timing_decorator
    def trapezoid_rule(self):
        """
        Метод трапеций для интегрирования.

        Возвращает:
        Результат численного интегрирования.

        """
        return IM.trapezoid_rule(func=self.func, a=self.a, b=self.b, n=self.n)
    
    @timing_decorator
    def simpson_rule(self):
        """
        Метод Симпсона для интегрирования.

        Возвращает:
        Результат численного интегрирования.

        """
        return IM.simpson_rule(func=self.func, a=self.a, b=self.b, n=self.n)
    
    @timing_decorator
    def newton_rule(self):
        """
        Метод Ньютона для интегрирования.

        Возвращает:
        Результат численного интегрирования.

        """
        return IM.newton_rule(func=self.func, a=self.a, b=self.b, n=self.n)

    
class NonlinearEquations:
    def __init__(self, a: float, b: float, eps: float, func):
        '''
        Инициализирует класс NonlinearEquations.

        Аргументы:
            a (float): Левая граница интервала.
            b (float): Правая граница интервала.
            eps (float): Допустимая погрешность решения.
            func (function): Функция, представляющая нелинейное уравнение.
        '''
        self.a = a
        self.b = b
        self.epsilon = eps
        self.func = func

    @timing_decorator
    def division(self):    
        '''
        Решает нелинейное уравнение методом деления отрезка пополам.

        Возвращает:
            Приближенное решение нелинейного уравнения.
        '''
        return NLE.division(func=self.func, a=self.a, b=self.b, epsilon=self.epsilon)
    
    @timing_decorator
    def chords_method(self):
        '''
        Решает нелинейное уравнение методом хорд.

        Возвращает:
            Приближенное решение нелинейного уравнения.
        '''
        return NLE.chords_method(func=self.func, a=self.a, b=self.b, epsilon=self.epsilon)

    @timing_decorator
    def newton_method(self):
        '''
        Решает нелинейное уравнение методом Ньютона.

        Возвращает:
            Приближенное решение нелинейного уравнения.
        '''
        return NLE.newton_method(func=self.func, a=self.a, b=self.b, epsilon=self.epsilon)
    
    @timing_decorator
    def iteration_method(self, phi_func=None):
        '''
        Решает нелинейное уравнение методом простых итераций.

        Аргументы:
            phi_func (function, optional): Функция для метода итераций.
                Если не предоставлено, будет использоваться функция по умолчанию.

        Возвращает:
            Приближенное решение нелинейного уравнения.
        '''
        return NLE.iteration_method(func=self.func, a=self.a, b=self.b, epsilon=self.epsilon, phi_func=phi_func)


class LinearEquationsMethods:
    def __init__(self, a_matrix=None, b_matrix=None):
        if a_matrix is not None and b_matrix is not None:
            self.original_matrix = np.array(a_matrix)
            self.right_column = np.array(b_matrix)
        else:
            self.input_matrix_A()
            self.input_vector_B()

    def input_matrix_A(self):
        """
        Метод для ввода матрицы коэффициентов A.
        """
        self.original_matrix = LE.input_matrix_A()

    def input_vector_B(self):
        """
        Метод для ввода вектора правой части B.
        """
        self.right_column = LE.input_vector_B()
        
    @timing_decorator
    def gaussian_main_element_method(self):
        """
        Решает систему линейных уравнений с использованием метода Гаусса с частичной перестановкой (LU-разложение).

        Returns:
            np.array: Решение системы линейных уравнений.
        """
        L, U = LE.lu_decomposition(self.original_matrix)
        solutions = LE.lu_solve(L, U, self.right_column)
        return solutions
    
    @timing_decorator
    def run_method(self):
        """
        Решает систему линейных уравнений методом Гаусса.

        Returns:
            np.array: Решение системы линейных уравнений.
        """
        solutions = LE.run_method(matrix=self.original_matrix, vector=self.right_column)
        return solutions
    
    @timing_decorator
    def gaussian_method(self):
        """
        Решает систему линейных уравнений методом Гаусса.

        Returns:
            np.array: Решение системы линейных уравнений.
        """
        solutions = LE.gaussian_method(
            a_matrix=self.original_matrix, 
            b_matrix=self.right_column
        )
        return solutions

    @timing_decorator
    def determinant(self):
        """
        Метод для вычисления определителя матрицы A.
        Returns:
            float: Определитель матрицы A.
        """
        det = np.linalg.det(self.original_matrix)
        return det

    @timing_decorator
    def inverse_matrix(self):
        """
        Метод для вычисления обратной матрицы A.
        Returns:
            np.array: Обратная матрица A^(-1).
        """
        inv_matrix = LE.inverse_matrix_method(A=self.original_matrix)
        return inv_matrix
    
    
    # импортирование метода зейделя и якоби с использованием timing decorator, дать описание в документации
    @timing_decorator
    def jacobi_method(self):
        """
        Решает систему линейных уравнений методом Якоби.

        Returns:
            np.array: Решение системы линейных уравнений.
        """
        solutions = LE.jacobi_method(
            a_matrix=self.original_matrix, 
            b_matrix=self.right_column
        )
        return solutions
    
    
    @timing_decorator
    def gauss_seidel_method(self):
        """
        Решает систему линейных уравнений методом Зейделя.

        Returns:
            np.array: Решение системы линейных уравнений.
        """
        solutions = LE.gauss_seidel_method(
            a_matrix=self.original_matrix, 
            b_matrix=self.right_column
        )
        return solutions