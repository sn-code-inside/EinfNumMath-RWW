import numpy as np


def rueckwaerts_einsetzen(R, b):
    assert (len(b.shape) == 1), 'Rechte seite ist kein Vektor'
    n = b.shape[0]
    assert (R.shape == (n, n)), 'Matrix hat falsche Dimensionen'

    x = np.empty_like(b)

    x[n - 1] = b[n - 1] / R[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        xr = 0
        for j in range(i + 1, n):
            xr += R[i, j] * x[j]
        x[i] = (b[i] - xr) / R[i, i]
    return x


def LR_zerlegung_einfach(A):
    assert (A.shape[0] == A.shape[1]), 'Matrix ist nicht Quadratisch'
    n = A.shape[0]

    for i in range(0, n):
        for k in range(i + 1, n):
            A[k, i] = A[k, i] / A[i, i]
            for j in range(i + 1, n):
                A[k, j] = A[k, j] - A[k, i] * A[i, j]
    return None


def vorwaerts_einsetzen_ohne_diag(L, b):
    # Wir nehmen an das alle Diagonaleinträge 1 sind um die modifizierte
    # Matrix ohne extra speicher verwenden zu können
    x = np.zeros_like(b)

    for i in range(0, b.shape[0]):
        xr = 0
        for j in range(0, i):
            xr += L[i, j] * x[j]
        x[i] = (b[i] - xr)
    return x


def LR_zerlegung_mit_pivot(A):
    assert (A.shape[0] == A.shape[1]), 'Matrix ist nicht Quadratisch'
    n = A.shape[0]
    pivot = []

    for i in range(0, n):
        # Wir suchen das Pivotelement und vertauschen die Zeilen.
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
