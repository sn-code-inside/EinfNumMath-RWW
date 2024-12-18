import numpy as np


def qr_givens(A):
    n, m = A.shape
    QT = np.identity(n, dtype=A.dtype)

    for i in range(m):
        for j in range(i + 1, n):
            c, s = A[i, i], -A[j, i]
            nrm = np.sqrt(c**2 + s**2)
            c, s = c / nrm, s / nrm
            for k in range(i, m):
                t1, t2 = A[i, k], A[j, k]
                A[i, k] = c * t1 - s * t2
                A[j, k] = s * t1 + c * t2

            for k in range(n):
                t1, t2 = QT[i, k], QT[j, k]
                QT[i, k] = c * t1 - s * t2
                QT[j, k] = s * t1 + c * t2
    return QT
