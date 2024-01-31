


def lagrange_interpolation(x_data, y_data):
    from sympy import symbols, simplify
    x = symbols('x')
    n = len(x_data)
    result = 0

    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term

    return simplify(result)

def aitken_interpolation(x_data, y_data):
    from sympy import symbols, simplify
    x = symbols('x')
    n = len(x_data)
    a = [0] * n

    for i in range(n):
        a[i] = y_data[i]

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            a[i] = ((x - x_data[i - j]) * a[i] - (x - x_data[i]) * a[i - 1]) / (x_data[i] - x_data[i - j])

    return simplify(a[n - 1])


def cubic_spline(x_data, y_data):
    from scipy.interpolate import CubicSpline
    from sympy import symbols, lambdify
    cs = CubicSpline(x_data, y_data)
    f = lambda x: cs(x)
    return f