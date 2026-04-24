import numpy as np


def f1(x):
    return x ** 2

def f2(x):
    return np.sin(x)

def f3(x):
    return np.exp(x)

def f4(x):
    return 1 / (1 + x ** 2)

def F1(x):
    return (x ** 3) / 3

def F2(x):
    return -np.cos(x)

def F3(x):
    return np.exp(x)

def F4(x):
    return np.arctan(x)

functions = [
    {"name": "x^2", "f": f1, "F": F1, "a": 0, "b": 3},
    {"name": "sin(x)", "f": f2, "F": F2, "a": 0, "b": np.pi / 2},
    {"name": "exp(x)", "f": f3, "F": F3, "a": 0, "b": 1},
    {"name": "1/(1+x^2)", "f": f4, "F": F4, "a": 0, "b": 1}
]

def change_variable(f, a, b):
    #Возвращает функцию g(t) = f((a+b)/2 + (b-a)/2 * t) для перехода к [-1,1]
    def g(t):
        x = (a + b) / 2 + (b - a) / 2 * t
        return f(x)
    return g

def finite_difference(f, x, h=1e-6, order=1):
    #Численное вычисление производной (для формулы Эйлера-Маклорена)
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    elif order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
    elif order == 3:
        return (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h ** 3)
    else:
        raise ValueError("Поддерживаются производные до 3-го порядка")

def left_rectangle(f, a, b, n):
    h = (b - a) / n
    result = 0.0
    for i in range(n):
        result += f(a + i * h)
    return result * h

def right_rectangle(f, a, b, n):
    h = (b - a) / n
    result = 0.0
    for i in range(1, n + 1):
        result += f(a + i * h)
    return result * h

def middle_rectangle(f, a, b, n):
    h = (b - a) / n
    result = 0.0
    for i in range(n):
        result += f(a + i * h + h / 2)
    return result * h

def trapezia(f, a, b, n):
    h = (b - a) / n
    result = (f(a) + f(b)) / 2
    for i in range(1, n):
        result += f(a + i * h)
    return result * h

def simpson(f, a, b, n):
    if n % 2 != 0:
        n += 1  # Симпсон требует чётного числа отрезков
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 0:
            result += 2 * f(a + i * h)
        else:
            result += 4 * f(a + i * h)
    return result * h / 3

def three_eighths(f, a, b, n):
    if n % 3 != 0:
        n = n + (3 - n % 3) if n % 3 != 0 else n
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * f(a + i * h)
        else:
            result += 3 * f(a + i * h)
    return result * 3 * h / 8

gauss_nodes_weights = {
    2: ([-0.5773502691896257, 0.5773502691896257], [1.0, 1.0]),
    3: ([-0.7745966692414834, 0.0, 0.7745966692414834], [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
    4: ([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
        [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]),
    5: ([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
        [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891])
}

def gauss_quadrature(f, a, b, n):
    nodes, weights = gauss_nodes_weights[n]
    g = change_variable(f, a, b)
    result = sum(w * g(t) for w, t in zip(weights, nodes))
    return (b - a) / 2 * result

chebyshev_nodes = {
    2: [-0.577350, 0.577350],
    3: [-0.707107, 0.0, 0.707107],
    4: [-0.794654, -0.187592, 0.187592, 0.794654],
    5: [-0.832497, -0.374541, 0.0, 0.374541, 0.832497]
}

def chebyshev_quadrature(f, a, b, n):
    nodes = chebyshev_nodes[n]
    weight = 2.0 / n
    g = change_variable(f, a, b)
    result = sum(g(t) for t in nodes)
    return (b - a) / 2 * weight * result

rado_nodes_weights = {
    2: ([-1 / 3, 1], [1.0, 1.0]),
    3: ([-0.289897, 0.689898, 1.0], [0.277778, 0.722222, 0.277778]),
    4: ([-0.575319, 0.181066, 0.822824, 1.0], [0.163266, 0.466565, 0.370169, 0.122222]),
    5: ([-0.720480, -0.167181, 0.446314, 0.885792, 1.0], [0.104656, 0.337433, 0.410087, 0.147824, 0.074074])
}

def radau_quadrature(f, a, b, n):
    nodes, weights = rado_nodes_weights[n]
    g = change_variable(f, a, b)
    result = sum(w * g(t) for w, t in zip(weights, nodes))
    return (b - a) / 2 * result

lobatto_nodes_weights = {
    3: ([-1.0, 0.0, 1.0], [1 / 3, 4 / 3, 1 / 3]),
    4: ([-1.0, -0.4472135954999579, 0.4472135954999579, 1.0],
        [1 / 6, 5 / 6, 5 / 6, 1 / 6]),
    5: ([-1.0, -0.6546536707079771, 0.0, 0.6546536707079771, 1.0],
        [0.1, 0.5444444444444444, 0.7111111111111111, 0.5444444444444444, 0.1])
}

def lobatto_quadrature(f, a, b, n):
    nodes, weights = lobatto_nodes_weights[n]
    g = change_variable(f, a, b)
    result = sum(w * g(t) for w, t in zip(weights, nodes))
    return (b - a) / 2 * result

laguerre_nodes_weights = {
    2: ([0.585786437626905, 3.414213562373095], [0.8535533905932737, 0.1464466094067263]),
    3: ([0.4157745567834791, 2.294280360279042, 6.289945082937479],
        [0.711093009929173, 0.2785177335692408, 0.0103892565015863]),
    4: ([0.3225476896193923, 1.7457611011583466, 4.536620296921128, 9.395070912301133],
        [0.6031541043416336, 0.3574186924377997, 0.0388879085150054, 0.0005392947055613]),
    5: ([0.2635603197181409, 1.413403059106517, 3.596425771040722, 7.085810005858837, 12.640800844275782],
        [0.5217556105828086, 0.3986668110831767, 0.0759424496817076, 0.0036117586799223, 0.0000233699723859])
}

def laguerre_quadrature(f, n):
    nodes, weights = laguerre_nodes_weights[n]
    return sum(w * f(x) for w, x in zip(weights, nodes))

hermite_nodes_weights = {
    2: ([-0.7071067811865476, 0.7071067811865476], [0.8862269254527579, 0.8862269254527579]),
    3: ([-1.224744871391589, 0.0, 1.224744871391589], [0.2954089751509193, 1.181635900603677, 0.2954089751509193]),
    4: ([-1.6506801238857855, -0.5246476232752903, 0.5246476232752903, 1.6506801238857855],
        [0.0813128354472458, 0.8049140900055128, 0.8049140900055128, 0.0813128354472458]),
    5: ([-2.020182870456086, -0.9585724646138185, 0.0, 0.9585724646138185, 2.020182870456086],
        [0.0199532420590459, 0.3936193231522412, 0.9453087204829419, 0.3936193231522412, 0.0199532420590459])
}

def hermite_quadrature(f, n):
    nodes, weights = hermite_nodes_weights[n]
    return sum(w * f(x) for w, x in zip(weights, nodes))

def euler_maclaurin(f, a, b, n, include_derivatives=True, derivative_order=1):
    h = (b - a) / n
    result = (f(a) + f(b)) / 2
    for i in range(1, n):
        result += f(a + i * h)
    result *= h

    if include_derivatives:
        if derivative_order >= 1:
            df_a = finite_difference(f, a, h=1e-6, order=1)
            df_b = finite_difference(f, b, h=1e-6, order=1)
            result += (h ** 2 / 12) * (df_a - df_b)

        if derivative_order >= 2:
            d3f_a = finite_difference(f, a, h=1e-4, order=3)
            d3f_b = finite_difference(f, b, h=1e-4, order=3)
            result -= (h ** 4 / 720) * (d3f_a - d3f_b)

    return result



print("1. Квадратуры прямоугольников, трапеции, Симпсона, 3/8")

n_values = [3, 4, 6, 8]

for func in functions:
    print(f"\nФУНКЦИЯ: {func['name']} на отрезке [{func['a']}, {func['b']}]")
    exact = func["F"](func["b"]) - func["F"](func["a"])
    print(f"Точное значение: {exact:.12f}")

    for n in n_values:
        print(f"\nn = {n} (число отрезков):")

        I_left = left_rectangle(func["f"], func["a"], func["b"], n)
        I_right = right_rectangle(func["f"], func["a"], func["b"], n)
        I_mid = middle_rectangle(func["f"], func["a"], func["b"], n)
        I_trap = trapezia(func["f"], func["a"], func["b"], n)
        I_simp = simpson(func["f"], func["a"], func["b"], n)

        print(f"  Левые прямоугольники:     {I_left:.10f}  | погрешность = {abs(I_left - exact):.2e}")
        print(f"  Правые прямоугольники:    {I_right:.10f}  | погрешность = {abs(I_right - exact):.2e}")
        print(f"  Средние прямоугольники:   {I_mid:.10f}  | погрешность = {abs(I_mid - exact):.2e}")
        print(f"  Трапеций:                 {I_trap:.10f}  | погрешность = {abs(I_trap - exact):.2e}")
        print(f"  Симпсона:                 {I_simp:.10f}  | погрешность = {abs(I_simp - exact):.2e}")

        if n % 3 == 0:
            I_38 = three_eighths(func["f"], func["a"], func["b"], n)
            print(f"  Трёх восьмых:             {I_38:.10f}  | погрешность = {abs(I_38 - exact):.2e}")

print("\n2. Квадратуры Гаусса, Чебышева, Радо, Лобатто")

n_nodes = [2, 3, 4, 5]

for func in functions:
    print(f"\nФУНКЦИЯ: {func['name']} на отрезке [{func['a']}, {func['b']}]")
    exact = func["F"](func["b"]) - func["F"](func["a"])
    print(f"Точное значение: {exact:.12f}")

    for n in n_nodes:
        print(f"\nn = {n} (число узлов):")

        try:
            I_gauss = gauss_quadrature(func["f"], func["a"], func["b"], n)
            print(f"  Гаусс (порядок {2 * n - 1}):   {I_gauss:.12f}  | погрешность = {abs(I_gauss - exact):.2e}")
        except ValueError as e:
            print(f"  Гаусс: {e}")

        try:
            I_cheb = chebyshev_quadrature(func["f"], func["a"], func["b"], n)
            print(f"  Чебышев:                  {I_cheb:.12f}  | погрешность = {abs(I_cheb - exact):.2e}")
        except ValueError as e:
            print(f"  Чебышев: {e}")

        try:
            I_radau = radau_quadrature(func["f"], func["a"], func["b"], n)
            print(f"  Радо (фикс. правый узел): {I_radau:.12f}  | погрешность = {abs(I_radau - exact):.2e}")
        except ValueError as e:
            print(f"  Радо: {e}")

        if n >= 3:
            try:
                I_lobatto = lobatto_quadrature(func["f"], func["a"], func["b"], n)
                print(f"  Лобатто (фикс. оба конца): {I_lobatto:.12f}  | погрешность = {abs(I_lobatto - exact):.2e}")
            except ValueError as e:
                print(f"  Лобатто: {e}")

print("\n3. Формула Эйлера-Маклорена (сравнение с трапецией)")

for func in functions:
    print(f"\nФУНКЦИЯ: {func['name']} на отрезке [{func['a']}, {func['b']}]")
    exact = func["F"](func["b"]) - func["F"](func["a"])
    print(f"Точное значение: {exact:.12f}")

    for n in [4, 8, 16]:
        I_trap = trapezia(func["f"], func["a"], func["b"], n)
        I_em1 = euler_maclaurin(func["f"], func["a"], func["b"], n, include_derivatives=True, derivative_order=1)
        I_em2 = euler_maclaurin(func["f"], func["a"], func["b"], n, include_derivatives=True, derivative_order=2)

        print(f"\nn = {n}:")
        print(f"  Трапеция:             {I_trap:.12f}  | погрешность = {abs(I_trap - exact):.2e}")
        print(f"  Эйлер-Маклорен (h^2): {I_em1:.12f}  | погрешность = {abs(I_em1 - exact):.2e}")
        print(f"  Эйлер-Маклорен (h^4): {I_em2:.12f}  | погрешность = {abs(I_em2 - exact):.2e}")

print("\n4. Квадратуры Гаусса-Лагерра и Гаусса-Эрмита (интегралы с весом)")

print("\n4.1. Квадратура Гаусса-Лагерра: ФУНКЦИЯ sin(x), точное значение = 0.5")
for n in [2, 3, 4, 5]:
    I_laguerre = laguerre_quadrature(np.sin, n)
    print(f"  n = {n}: {I_laguerre:.10f}  | погрешность = {abs(I_laguerre - 0.5):.2e}")

exact_hermite = (3/4) * np.sqrt(np.pi)
print(f"\n4.2. Квадратура Гаусса-Эрмита: ФУНКЦИЯ - х^4, точное значение = {exact_hermite:.10f}")
for n in [2, 3, 4, 5]:
    I_hermite = hermite_quadrature(lambda x: x ** 4, n)
    print(f"  n = {n}: {I_hermite:.10f}  | погрешность = {abs(I_hermite - exact_hermite):.2e}")
