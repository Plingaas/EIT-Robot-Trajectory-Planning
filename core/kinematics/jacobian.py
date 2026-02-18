import numpy as np

from core.se3 import adjoint
from core.types import Matrix4x4, Matrix6xn, Vectorn
from core.kinematics.fk import fk_all
from core.kinematics.validation import validate_poe_inputs

def jacobian(M: Matrix4x4, S: Matrix6xn, q: Vectorn) -> Matrix6xn:
    """
    Space Jacobian J_s(q) for the space-frame PoE model:
        T(q) = exp([S1]q1) ... exp([Sn]qn) M

    Returns:
        J : (6,n) space Jacobian
            J[:,0] = S[:,0]
            J[:,i] = Adjoint(A_i) @ S[:,i]   for i = 1..n-1
        where A_i = exp([S1]q1)...exp([Si]qi) and A_0 = I.
    """
    M = np.asarray(M, dtype=float)
    S = np.asarray(S, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    validate_poe_inputs(M, S, q)

    n = q.size
    A_list, _ = fk_all(M, S, q)  

    J = np.zeros((6, n), dtype=float)
    J[:, 0] = S[:, 0]

    for i in range(1, n):
        adjoint_A_i = adjoint(A_list[i])
        J_i = adjoint_A_i @ S[:, i]
        J[:, i] = J_i
    return J
