import math

def left_rectangular(func, a, b, n):
    '''
    Вычисляет интеграл функции методом левых прямоугольников.

    Аргументы:
    - func: Функция, которую нужно проинтегрировать.
    - a: Нижний предел интегрирования.
    - b: Верхний предел интегрирования.
    - n: Количество разбиений.

    Возвращает:
    Результат численного интегрирования методом левых прямоугольников.
    '''
    result = 0
    x = a
    summator = 0
    dx = math.fabs(b - a) / n
    while x <= b:
        summator += func(x)
        x += dx
    result = (b - a) / n * summator
    return result

def right_rectangular(func, a, b, n):
    '''
    Вычисляет интеграл функции методом правых прямоугольников.

    Аргументы:
    - func: Функция, которую нужно проинтегрировать.
    - a: Нижний предел интегрирования.
    - b: Верхний предел интегрирования.
    - n: Количество разбиений.

    Возвращает:
    Результат численного интегрирования методом правых прямоугольников.
    '''
    result = 0
    x = b
    summator = 0
    dx = math.fabs(b - a) / n
    while x >= a:
        summator += func(x)
        x -= dx
    result = (b - a) / n * summator
    return result

def central_rectangular(func, a, b, n):
    '''
    Вычисляет интеграл функции методом центральных прямоугольников.

    Аргументы:
    - func: Функция, которую нужно проинтегрировать.
    - a: Нижний предел интегрирования.
    - b: Верхний предел интегрирования.
    - n: Количество разбиений.

    Возвращает:
    Результат численного интегрирования методом центральных прямоугольников.
    '''
    result = 0
    x = a
    summator = 0
    dx = math.fabs(b - a) / n
    while x <= b:
        summator += func(x - dx / 2)
        x += dx
    result = (b - a) / n * summator
    return result

def trapezoid_rule(func, a, b, n):
    '''
    Вычисляет интеграл функции методом трапеций.

    Аргументы:
    - func: Функция, которую нужно проинтегрировать.
    - a: Нижний предел интегрирования.
    - b: Верхний предел интегрирования.
    - n: Количество разбиений.

    Возвращает:
    Результат численного интегрирования методом трапеций.
    '''
    result = 0
    x = a
    summator = 0
    dx = math.fabs(b - a) / n
    while x <= b - dx:
        summator += func(x)
        x += dx
    result = summator + (func(a) + func(b)) / 2
    result = (b - a) / n * result
    return result

def simpson_rule(func, a, b, n):
    '''
    Вычисляет интеграл функции методом Симпсона.

    Аргументы:
    - func: Функция, которую нужно проинтегрировать.
    - a: Нижний предел интегрирования.
    - b: Верхний предел интегрирования.
    - n: Количество разбиений.

    Возвращает:
    Результат численного интегрирования методом Симпсона.
    '''
    x = a
    summator_1 = 0
    summator_2 = 0
    dx = math.fabs(b - a) / (2 * n)
    while x <= b:
        summator_1 += func(x - dx)
        summator_2 += func(x)
        x += dx * 2
    result = func(a) - func(b) + (4 * summator_1) + (2 * summator_2)
    return result
   

def newton_rule(func, a, b, n):
    '''
    Вычисляет интеграл функции методом Ньютона.

    Аргументы:
    - func: Функция, которую нужно проинтегрировать.
    - a: Нижний предел интегрирования.
    - b: Верхний предел интегрирования.
    - n: Количество разбиений.

    Возвращает:
    Результат численного интегрирования методом Ньютона.
    '''
    x = a
    summator_1 = 0
    summator_2 = 0
    dx = math.fabs(b - a) / (3 * n)
    while x <= b:
        summator_1 += func(x - dx / 3 * 3) + func(x - dx / 3)
        summator_2 += func(x)
        x += dx * 3
    result = func(a) - func(b) + (3 * summator_1) + (2 * summator_2)
    return (b - a) / (8 * n) * result
