import numpy as np 
from core.types import Matrix4x4, Matrix6xn, Vectorn
from core.se3 import exp_se3, hat
from core.kinematics.validation import validate_poe_inputs


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
    validate_poe_inputs(M, S, q)    

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
    validate_poe_inputs(M, S, q)

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


def fk_links(M_links: list[Matrix4x4], S: Matrix6xn, q: Vectorn) -> list[Matrix4x4]:
    """
    Space frame Product of Exponentials (PoE) for link frames.

    M_links : list of length n+1
        Home configuration for each link frame in the space frame.
        M_links[0] is the fixed base/mount frame, M_links[k] corresponds to the
        link after joint k.

    S : (6,n)
        Screw axes in the space frame.

    q : (n,)
        Joint angles.

    Returns:
        T_links : list of length n+1
            Link transforms T_k = A_k @ M_links[k], with A_0 = I.
    """
    if not M_links:
        raise ValueError("M_links must contain at least one home frame.")

    for M_link in M_links:
        validate_poe_inputs(M_link, S, q)

    if len(M_links) != np.asarray(q).size + 1:
        raise ValueError("M_links must have length q.size + 1.")

    home_frames = []
    for M_link in M_links:
        M_link = np.asarray(M_link, dtype=float)
        home_frames.append(M_link)

    A_list, _ = fk_all(home_frames[0], S, q)
    return [A_list[i] @ home_frames[i] for i in range(len(home_frames))]

