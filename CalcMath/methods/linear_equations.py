import numpy as np

def input_matrix_A():
    """
    Ввод матрицы коэффициентов А пользователем.

    Returns:
        np.array: Матрица коэффициентов А в виде массива NumPy.
    """
    n = int(input("Введите размерность матрицы коэффициентов A: "))
    original_matrix = []
    print("Введите элементы матрицы коэффициентов A:")
    for i in range(n):
        row = [float(x) for x in input().split()]
        original_matrix.append(row)
    matrix_A = np.array(original_matrix)
    return matrix_A


def input_vector_B():
    """
    Ввод правой части системы уравнений B пользователем.

    Returns:
        np.array: Вектор правой части B в виде массива NumPy.
    """
    n = int(input("Введите размерность вектора правой части B: "))
    vector_B = []
    print("Введите элементы вектора правой части B:")
    for i in range(n):
        element = float(input())
        vector_B.append(element)
    vector_B = np.array(vector_B)
    return vector_B


def input_matrix():
    '''
    Ввод матрицы с клавиатуры.

    Returns:
        np.array: Матрица, представленная в виде массива NumPy.
    '''
    n = int(input("Введите размерность матрицы: "))
    original_matrix = []
    print("Введите элементы матрицы:")
    for i in range(n):
        row = [float(x) for x in input().split()]
        original_matrix.append(row)
    matrix = np.array(original_matrix)
    return matrix


def output_matrix(matrix):
    '''
    Вывод матрицы на экран.

    Args:
        matrix (np.array): Матрица, представленная в виде массива NumPy.
    '''
    for row in matrix:
        print(" ".join(str(x) for x in row))


def determinant(matrix):
    '''
    Вычисление определителя матрицы.

    Args:
        matrix (np.array): Матрица, представленная в виде массива NumPy.

    Returns:
        float: Определитель матрицы.
    '''
    return np.linalg.det(matrix)       
    

def gaussian_straight(matrix, vector):
    n = len(vector)

    for i in range(n):
        # Поиск максимального элемента в текущем столбце и перестановка строк
        max_row = i
        for j in range(i + 1, n):
            if abs(matrix[j][i]) > abs(matrix[max_row][i]):
                max_row = j

        # Обмен строками, чтобы максимальный элемент был на верхней позиции
        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
        vector[i], vector[max_row] = vector[max_row], vector[i]

        # Приведение матрицы к верхнетреугольному виду
        pivot = matrix[i][i]
        for j in range(i + 1, n):
            factor = matrix[j][i] / pivot
            vector[j] -= factor * vector[i]
            for k in range(i, n):
                matrix[j][k] -= factor * matrix[i][k]

    return matrix, vector


def gaussian_back(matrix, vector):
    n = len(vector)
    x = [0] * n

    for i in range(n - 1, -1, -1):
        x[i] = vector[i]
        for j in range(i + 1, n):
            x[i] -= matrix[i][j] * x[j]
        x[i] /= matrix[i][i]

    return x


def gaussian_method(a_matrix, b_matrix):
    '''
    Решение системы линейных уравнений методом Гаусса.

    Args:
        matrix (np.array): Матрица, представленная в виде массива NumPy.

    Returns:
        np.array: Решение системы линейных уравнений.
    '''
    mat, vec = gaussian_straight(matrix=a_matrix, vector=b_matrix)
    res_back = gaussian_back(matrix=mat, vector=vec)
    rounded_x = [int(round(val)) if abs(val - round(val)) < 0.000000001 else val for val in res_back]
    return rounded_x


def lu_decomposition(A):
    """
    Выполняет LU-разложение матрицы.

    Args:
        A (np.array): Исходная матрица.

    Returns:
        np.array: Матрица L (нижняя треугольная) и матрица U (верхняя треугольная).
    """
def lu_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i][i] = 1  # Диагональ элементов L равна 1

        for j in range(i, n):
            U[i][j] = matrix[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]

        for j in range(i + 1, n):
            L[j][i] = matrix[j][i]
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]
            L[j][i] /= U[i][i]

    return L, U

def lu_solve(L, U, b):
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    # Решение системы Ly = b методом прямого хода
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]

    # Решение системы Ux = y методом обратного хода
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]

    return x
    

def gaussian_main_element_method(A, B):
    """
    Решает систему уравнений LUx = b.

    Args:
        L (np.array): Матрица L (нижняя треугольная) из LU-разложения.
        U (np.array): Матрица U (верхняя треугольная) из LU-разложения.
        B (np.array): Правая часть системы.

    Returns:
        np.array: Решение системы уравнений.
    """
    L, U = lu_decomposition(A)
    Y = gaussian_method(L, B)
    x = gaussian_method(U, Y)
    return x


def determinant_method(A):
    """
    Рекурсивное вычисление определителя матрицы.

    Args:
        matrix (np.array): Матрица, представленная в виде массива NumPy.

    Returns:
        float: Определитель матрицы.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A должна быть квадратной.")
    
    # Создаем копию матрицы A, чтобы избежать изменения оригинальной матрицы
    matrix = A.copy()
    
    # Применяем метод Гаусса для преобразования матрицы в верхнюю треугольную форму
    sign = 1  # Множитель для учета перестановок строк или столбцов
    for i in range(n):
        if matrix[i, i] == 0:
            # Если элемент на главной диагонали равен 0, найдем ненулевой элемент в том же столбце и поменяем строки
            for j in range(i + 1, n):
                if matrix[j, i] != 0:
                    matrix[[i, j]] = matrix[[j, i]]  # Перестановка строк
                    sign *= -1
                    break
        for j in range(i + 1, n):
            factor = matrix[j, i] / matrix[i, i]
            matrix[j, i:] -= factor * matrix[i, i:]
    
    # Вычисляем определитель как произведение элементов на главной диагонали и множителя sign
    det = sign
    for i in range(n):
        det *= matrix[i, i]
    
    return det


def inverse_matrix_method(A):
    """
    Вычисление обратной матрицы методом Гаусса.

    Args:
        A (np.array): Исходная матрица.

    Returns:
        np.array: Обратная матрица A^(-1).
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A должна быть квадратной.")
    extended_matrix = np.hstack([A, np.eye(n)])
    for i in range(n):
        if extended_matrix[i, i] == 0:
            for j in range(i + 1, n):
                if extended_matrix[j, i] != 0:
                    extended_matrix[[i, j]] = extended_matrix[[j, i]]  
                    break
        pivot = extended_matrix[i, i]
        extended_matrix[i, :] /= pivot
        for j in range(n):
            if j != i:
                factor = extended_matrix[j, i]
                extended_matrix[j, :] -= factor * extended_matrix[i, :]
    
    # Извлекаем обратную матрицу из расширенной матрицы
    inverse_A = extended_matrix[:, n:]
    
    return inverse_A


def run_method(matrix, vector):
    n = len(matrix)
    c = matrix[0][1]
    b = matrix[0][0]
    d = vector[0]
    alpha = [0] * n
    beta = [0] * n
    alpha[0] = -c / b
    beta[0] = -d / b

    for i in range(1, n-1):
        b = matrix[i][i]
        a = matrix[i][i - 1]
        c = matrix[i][i + 1] if i + 1 < n else 0
        d = vector[i]
        gamma = b + a * alpha[i - 1]
        alpha[i] = -c / gamma
        beta[i] = -(d - a * beta[i - 1]) / gamma
    
    d = vector[n-1]
    a = matrix[n-1][n-2]
    b = matrix[n-1][n-1]
    beta[n-1] = (d - a * beta[n-2]) / (b + a * alpha[n-2])
    
    x = np.zeros(n)
    x[n-1] = beta[n-1]
    for i in range(n-2, 0, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]

    return x


def jacobi_method(a_matrix, b_matrix, max_iterations=100, tolerance=1e-6):
    n = len(a_matrix)
    x = np.zeros(n)
    iteration = 0
    
    while iteration < max_iterations:
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = (b_matrix[i] - np.dot(a_matrix[i, :i], x[:i]) - np.dot(a_matrix[i, i+1:], x[i+1:])) / a_matrix[i, i]
        if np.max(np.abs(x_new - x)) < tolerance:
            print("The iterative method has diverged.")
            break
        x = x_new
        iteration += 1
    else:
        print("The iterative method has diverged after {} iterations.".format(max_iterations))
    
    return x


def gauss_seidel_method(a_matrix, b_matrix, max_iterations=100, tolerance=1e-6):
    a_matrix = np.array(a_matrix)
    n = len(a_matrix)
    x = np.zeros(n)
    iteration = 0
    
    while iteration < max_iterations:
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = (b_matrix[i] - np.dot(a_matrix[i, :i], x_new[:i]) - np.dot(a_matrix[i, i+1:], x[i+1:])) / a_matrix[i, i]
        if np.max(np.abs(x_new - x)) < tolerance:
            break
        x = x_new
        iteration += 1
    else:
        print("The iterative method has diverged after {} iterations.".format(max_iterations))
    
    return x