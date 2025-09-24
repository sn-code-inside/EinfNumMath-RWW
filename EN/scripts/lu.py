import numpy as np

__all__ = ['backward', 'forward', 'forward2', 'lu_pivot', 'cholesky']


def backward(U, b):
    assert (len(b.shape) == 1), 'rhs is not a vector'
    n = b.shape[0]
    assert (U.shape == (n, n)), "Matrix dimensions don't match"

    x = np.empty_like(b)

    x[n - 1] = b[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        xr = 0
        for j in range(i + 1, n):
            xr += U[i, j] * x[j]
        x[i] = (b[i] - xr) / U[i, i]
    return x


def forward(L, b):
    # We assume L has values 1 on the diagonal
    x = np.zeros_like(b)

    for i in range(0, b.shape[0]):
        xr = 0
        for j in range(0, i):
            xr += L[i, j] * x[j]
        x[i] = (b[i] - xr)
    return x


def forward2(L, b):
    x = np.zeros_like(b)

    for i in range(0, b.shape[0]):
        xr = 0
        for j in range(0, i):
            xr += L[i, j] * x[j]
        x[i] = (b[i] - xr) / L[i, i]
    return x


def lu_pivot(A):
    assert (A.shape[0] == A.shape[1]), 'Matrix ist nicht Quadratisch'
    n = A.shape[0]
    pivot = []

    for i in range(0, n):
        # Search for the pivot element and swap rows
        k = i
        for j in range(i, n):
            if abs(A[j, i]) > abs(A[k, i]):
                k = j
        A[[i, k], :] = A[[k, i], :]
        pivot.append([i, k])

        for k in range(i + 1, n):
            A[k, i] = A[k, i] / A[i, i]
            for j in range(i + 1, n):
                A[k, j] = A[k, j] - A[k, i] * A[i, j]

    return pivot


def cholesky(A):
    n, m = A.shape
    assert n == m, "Matrix dimensions don't match"
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
