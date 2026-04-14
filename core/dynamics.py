import numpy as np
from dataclasses import dataclass

from core.se3 import adjoint, exp_se3_twist, inv_se3
from core.so3 import skew
from core.types import Matrix4x4, Matrix6x6, Matrix6xn, Vector3, Vector6, Vectorn


def ad(V: Vector6) -> Matrix6x6:
    """Return the spatial cross-product operator ad(V) for twists V = [w, v]."""
    V = np.asarray(V, dtype=float).reshape(6,)
    w = V[:3]
    v = V[3:]

    ad_V = np.zeros((6, 6), dtype=float)
    ad_V[:3, :3] = skew(w)
    ad_V[3:, :3] = skew(v)
    ad_V[3:, 3:] = skew(w)
    return ad_V


def _relative_mlist(M_LIST: tuple[Matrix4x4, ...] | list[Matrix4x4]) -> list[Matrix4x4]:
    """Build consecutive home transforms from space-frame home poses."""
    relative = []
    for i in range(len(M_LIST) - 1):
        M_prev = np.asarray(M_LIST[i], dtype=float)
        M_next = np.asarray(M_LIST[i + 1], dtype=float)
        relative.append(inv_se3(M_prev) @ M_next)

    relative.append(np.eye(4, dtype=float))
    return relative


@dataclass(frozen=True)
class InverseDynamicsConstants:
    A: np.ndarray
    M_rel_inv: tuple[np.ndarray, ...]
    Ad_T_end: np.ndarray
    G_list: tuple[np.ndarray, ...]


def precompute_inverse_dynamics_constants(
    M_LIST: tuple[Matrix4x4, ...] | list[Matrix4x4],
    G_LIST: tuple[Matrix6x6, ...] | list[Matrix6x6],
    S: Matrix6xn,
) -> InverseDynamicsConstants:
    S = np.asarray(S, dtype=float)
    n = S.shape[1]

    if len(M_LIST) != n + 1:
        raise ValueError("M_LIST must contain n + 1 home frames.")
    if len(G_LIST) != n:
        raise ValueError("G_LIST must contain n spatial inertia matrices.")

    M_rel = _relative_mlist(M_LIST)
    M_rel_inv = tuple(inv_se3(np.asarray(M, dtype=float)) for M in M_rel)

    A = np.zeros((6, n), dtype=float)
    M_cumulative = np.eye(4, dtype=float)
    for i in range(n):
        M_cumulative = M_cumulative @ np.asarray(M_rel[i], dtype=float)
        A[:, i] = adjoint(inv_se3(M_cumulative)) @ S[:, i]

    return InverseDynamicsConstants(
        A=A,
        M_rel_inv=M_rel_inv,
        Ad_T_end=adjoint(M_rel_inv[n]),
        G_list=tuple(np.asarray(G_i, dtype=float) for G_i in G_LIST),
    )


def inverse_dynamics(
    q: Vectorn,
    q_dot: Vectorn,
    q_dot_dot: Vectorn,
    g: Vector3,
    Ftip: Vector6,
    M_LIST: tuple[Matrix4x4],
    G_LIST: tuple[Matrix6x6],
    S: Matrix6xn,
    constants: InverseDynamicsConstants | None = None,
) -> Vectorn:
    """
    Compute joint torques with recursive Newton-Euler inverse dynamics.

    Assumptions:
    - twists are ordered as [w, v]
    - M_LIST contains link home poses in the space frame, including the fixed base frame
    - Ftip is expressed in the final link / end-effector frame
    - g is expressed in the space frame and must use the same length units as M_LIST
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    q_dot = np.asarray(q_dot, dtype=float).reshape(-1)
    q_dot_dot = np.asarray(q_dot_dot, dtype=float).reshape(-1)
    g = np.asarray(g, dtype=float).reshape(3,)
    Ftip = np.asarray(Ftip, dtype=float).reshape(6,)
    S = np.asarray(S, dtype=float)

    n = q.size
    if q_dot.size != n or q_dot_dot.size != n:
        raise ValueError("q, q_dot, and q_dot_dot must have the same length.")
    if S.shape != (6, n):
        raise ValueError("S must have shape (6, n).")
    if len(M_LIST) != n + 1:
        raise ValueError("M_LIST must contain n + 1 home frames.")
    if len(G_LIST) != n:
        raise ValueError("G_LIST must contain n spatial inertia matrices.")

    if constants is None:
        constants = precompute_inverse_dynamics_constants(M_LIST=M_LIST, G_LIST=G_LIST, S=S)

    A = np.zeros((6, n), dtype=float)
    Ad_T = [np.eye(6, dtype=float) for _ in range(n + 1)]
    V = np.zeros((6, n + 1), dtype=float)
    V_dot = np.zeros((6, n + 1), dtype=float)
    V_dot[:, 0] = np.hstack((np.zeros(3, dtype=float), -g))
    tau = np.zeros(n, dtype=float)

    for i in range(n):
        A[:, i] = constants.A[:, i]

        joint_transform = exp_se3_twist(A[:, i], -q[i]) @ constants.M_rel_inv[i]
        Ad_T[i] = adjoint(joint_transform)

        V[:, i + 1] = Ad_T[i] @ V[:, i] + A[:, i] * q_dot[i]
        V_dot[:, i + 1] = (
            Ad_T[i] @ V_dot[:, i]
            + A[:, i] * q_dot_dot[i]
            + ad(V[:, i + 1]) @ (A[:, i] * q_dot[i])
        )

    Ad_T[n] = constants.Ad_T_end
    F = Ftip.copy()

    for i in range(n - 1, -1, -1):
        G_i = constants.G_list[i]
        F = (
            Ad_T[i + 1].T @ F
            + G_i @ V_dot[:, i + 1]
            - ad(V[:, i + 1]).T @ (G_i @ V[:, i + 1])
        )
        tau[i] = F @ A[:, i]

    return tau

# TODO; test inverse dynamics
