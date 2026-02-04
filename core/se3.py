import numpy as np

from scipy.linalg import expm, logm

from core.types import Matrix3x3, Matrix4x4, Matrix6x6, Vector3, Vector6
from core.so3 import skew

def homogeneous(R: Matrix3x3, p: Vector3) -> Matrix4x4:
    """Build 4x4 homogeneous transform ∈ SE(3) from R ∈ SO(3) and p ∈ ℝ³"""
    R = np.asarray(R, dtype=float)
    p = np.asarray(p, dtype=float)

    if R.shape != (3, 3):
        raise ValueError("homogeneous() expects R to be 3x3.")
    if p.size != 3:
        raise ValueError("homogeneous() expects p to have 3 elements.")

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p.reshape(3,)
    return T

def translate(x: float, y: float, z: float) -> Matrix4x4:
    """Return homogeneous transform ∈ SE(3) for translation by (x, y, z)."""
    return homogeneous(np.eye(3, dtype=float), np.array([x, y, z], dtype=float))


def inv_se3(T: Matrix4x4) -> Matrix4x4:
    """Return the inverse of a homogeneous transform ∈ SE(3)."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("inv_se3() expects a 4x4 matrix.")
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def hat(xi: Vector6) -> Matrix4x4:
    """
    Return an element of the Lie algebra se(3) as a 4x4 matrix Xî from
    a twist xi ∈ ℝ⁶ (xi = [ω, v]). This is NOT a homogeneous transform.
    """
    xi = np.asarray(xi, dtype=float).reshape(6,)
    w = xi[:3]
    v = xi[3:]

    Xi_hat = np.zeros((4, 4), dtype=float)
    Xi_hat[:3, :3] = skew(w)
    Xi_hat[:3, 3] = v
    return Xi_hat


def vee(Xi_hat: Matrix4x4) -> Vector6:
    """
    Return a twist xi ∈ ℝ⁶ (xi = [ω, v]) from an se(3) matrix Xî ∈ ℝ⁴×⁴.
    Inverse of the hat(·) operator.
    """
    Xi_hat = np.asarray(Xi_hat, dtype=float)
    if Xi_hat.shape != (4, 4):
        raise ValueError("vee() expects a 4x4 matrix.")
    W = Xi_hat[:3, :3]
    w = np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)
    v = Xi_hat[:3, 3]
    return np.hstack((w, v))


def adjoint(T: Matrix4x4) -> Matrix6x6:
    """Return Ad_T ∈ ℝ⁶ˣ⁶ for T ∈ SE(3), assuming twists are xi = [ω, v]."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("adjoint() expects a 4x4 matrix.")
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6), dtype=float)
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = skew(p) @ R
    return Ad

def exp_se3(Xi_hat: Matrix4x4) -> Matrix4x4:
    """Return T ∈ SE(3) as exp(Xî) for Xî ∈ se(3)."""
    Xi_hat = np.asarray(Xi_hat, dtype=float)
    if Xi_hat.shape != (4, 4):
        raise ValueError("exp_se3() expects a 4x4 matrix.")
    # TODO; Could implement from scratch later
    return expm(Xi_hat)


def exp_se3_twist(xi: Vector6, theta: float) -> Matrix4x4:
    """Return T ∈ SE(3) as exp(hat(xi)·theta) for twist xi = [ω, v]."""
    return expm(hat(xi) * theta)


def log_se3(T: Matrix4x4) -> Matrix4x4:
    """Return se(3) matrix Xî ∈ ℝ⁴ˣ⁴ as log(T) for T ∈ SE(3)."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("log_se3() expects a 4x4 matrix.")
    Xi = logm(T)
    Xi = np.real_if_close(Xi, tol=1000).astype(float)

    # Numerical cleanup
    Xi[3, :] = 0.0
    W = Xi[:3, :3]
    Xi[:3, :3] = 0.5 * (W - W.T)
    Xi[3, 3] = 0.0
    return Xi


