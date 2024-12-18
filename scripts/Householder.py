import numpy as np


def qr_householder(A):
    n, m = A.shape
    V = np.zeros_like(A)

    for i in range(m):
        V[i:, i] = A[i:, i]
        ei = np.zeros(n - i, dtype=A.dtype)
        ei[0] = 1.0
        V[i:, i] += np.sign(A[i, i]) * np.linalg.norm(V[i:, i]) * ei
        V[i:, i] /= np.linalg.norm(V[i:, i])
        for k in range(i, m):
            A[i:, k] -= 2 * np.inner(V[i:, i], A[i:, k]) * V[i:, i]
    return V
