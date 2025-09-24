import numpy as np

__all__ = ['neville', 'difference_1_forward', 'difference_1_central',
           'difference_2_forward', 'difference_2_central']


def neville(data, xi):
    n = data.shape[0]
    x = data[:, 0]
    p = np.diag(data[:, 1])
    
    for j in range(1, n):
        for k in range(n - j):
            p[k, k + j] = p[k, k + j - 1] + (xi - x[k]) * (p[k + 1, k + j] - p[k, k + j - 1]) / (x[k + j] - x[k])
    return p

def difference_1_forward(f, x, h):
    f0 = f(x)
    f1 = f(x + h)
    return (f1 - f0) / h


def difference_1_central(f, x, h):
    f0 = f(x - h)
    f1 = f(x + h)
    return (f1 - f0) / (2 * h)


def difference_2_forward(f, x, h):
    f0 = f(x)
    f1 = f(x + h)
    f2 = f(x + 2 * h)
    return (f2 - 2 * f1 + f0) / h**2


def difference_2_central(f, x, h):
    f0 = f(x - h)
    f1 = f(x)
    f2 = f(x + h)
    return (f2 - 2 * f1 + f0) / h**2