import numpy as np
from core.types import Matrix4x4, Matrix6xn, Vectorn

def validate_poe_inputs(M: Matrix4x4, S: Matrix6xn, q: Vectorn):
    M = np.asarray(M)
    S = np.asarray(S)
    q = np.asarray(q)

    if M.shape != (4, 4):
        raise ValueError("M must be 4x4.")

    if S.ndim != 2 or S.shape[0] != 6:
        raise ValueError("S must have shape (6, n).")

    if q.ndim != 1:
        raise ValueError("q must be 1D.")

    if S.shape[1] != q.size:
        raise ValueError("Mismatch between S columns and q size.")