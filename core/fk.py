import numpy as np 
from core.se3 import exp_se3, hat
from core.types import Matrix4x4, Matrix6xn, Vectorn

def _raise_on_invalid_input(M: Matrix4x4, S: Matrix6xn, q: Vectorn):
    if M.shape != (4, 4):
        raise ValueError("M must be 4x4.")
    if S.shape[0] != 6:
        raise ValueError("S must be 6xn.")
    if S.shape[1] != q.size:
        raise ValueError("Mismatch between S and q.")

def fk(M: Matrix4x4, S: Matrix6xn, q: Vectorn) -> Matrix4x4:
    """
    Space frame Product of Exponentials (PoE):
    M : (4,4) home configuration
    S : (6,n) screw axes in space frame
    q : (n,) joint angles

    Returns the homogeneous transform T_be ∈ SE(3) of the end-effector frame
    """
    M = np.asarray(M, dtype=float)
    S = np.asarray(S, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1) 
    _raise_on_invalid_input(M, S, q)    

    A = np.eye(4, dtype=float)
    for i in range(q.size):
        screw_axis = S[:, i]  
        twist_matrix = hat(screw_axis) 
        A = A @ exp_se3(twist_matrix * q[i])
    return A @ M


def fk_all(M: Matrix4x4, S: Matrix6xn, q: Vectorn) -> tuple[list[Matrix4x4], list[Matrix4x4]]:
    """
    Space frame Product of Exponentials (PoE) for all intermediate transforms:
    M : (4,4) home configuration
    S : (6,n) screw axes in space frame
    q : (n,) joint angles

    Returns:
        A_list : list of length n+1
            Partial products A_k = exp([S1]q1) ... exp([Sk]qk),
            with A_0 = I.

        T_list : list of length n+1
            End-effector transforms T_k = A_k @ M.
            In particular, T_list[-1] is the full forward kinematics result.
    """
    M = np.asarray(M, dtype=float)
    S = np.asarray(S, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    _raise_on_invalid_input(M, S, q)

    A = np.eye(4, dtype=float)
    A_list = [A.copy()]
    T_list = [M.copy()]

    for i in range(q.size):
        screw_axis = S[:, i]  
        twist_matrix = hat(screw_axis) 
        A = A @ exp_se3(twist_matrix * q[i])
        A_list.append(A.copy())
        T_list.append((A @ M).copy())
    return A_list, T_list
