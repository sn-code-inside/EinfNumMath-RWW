import numpy as np

def neville(data, xi):
    n = data.shape[0]
    x = data[:, 0]
    p = np.diag(data[:, 1])
    
    for j in range(1, n):
        for k in range(n - j):
            p[k, k + j] = p[k, k + j - 1] + (xi - x[k]) * (p[k + 1, k + j] - p[k, k + j - 1]) / (x[k + j] - x[k])
    return p


def differenz_einseitig_1(f, x, h):
    f0 = f(x)
    f1 = f(x + h)
    return (f1 - f0) / h


def differenz_zentral_1(f, x, h):
    f0 = f(x - h)
    f1 = f(x + h)
    return (f1 - f0) / (2 * h)


def differenz_einseitig_2(f, x, h):
    f0 = f(x)
    f1 = f(x + h)
    f2 = f(x + 2 * h)
    return (f2 - 2 * f1 + f0) / h**2


def differenz_zentral_2(f, x, h):
    f0 = f(x - h)
    f1 = f(x)
    f2 = f(x + h)
    return (f2 - 2 * f1 + f0) / h**2