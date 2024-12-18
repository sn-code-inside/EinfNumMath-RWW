import numpy as np


def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A)

    for j in range(0, n):
        l = 0
        for k in range(0, j):
            l += L[j, k]**2
        L[j, j] = np.sqrt(A[j, j] - l)

        for i in range(j + 1, n):
            l = 0
            for k in range(j):
                l += L[i, k] * L[j, k]
            L[i, j] = (A[i, j] - l) / L[j, j]
    return L


def vorwaerts_einsetzen(L, b):
    x = np.zeros_like(b)

    for i in range(0, b.shape[0]):
        xr = 0
        for j in range(0, i):
            xr += L[i, j] * x[j]
        x[i] = (b[i] - xr) / L[i, i]
    return x
