import numpy as np
from core.types import Matrix3x3, Vector3

def rotation_x(angle: float) -> Matrix3x3:
    """Return rotation matrix ∈ SO(3) for rotation about x-axis by angle (radians)."""
    cosa, sina = np.cos(angle), np.sin(angle)
    return np.array([
        [1,  0,   0],
        [0, cosa, -sina],
        [0, sina,  cosa]
    ], dtype=float)

def rotation_y(angle: float) -> Matrix3x3:
    """Return rotation matrix ∈ SO(3) for rotation about y-axis by angle (radians)."""
    cosa, sina = np.cos(angle), np.sin(angle)
    return np.array([
        [ cosa, 0, sina],
        [    0, 1,    0],
        [-sina, 0, cosa]
    ], dtype=float)

def rotation_z(angle: float) -> Matrix3x3:
    """Return rotation matrix ∈ SO(3) for rotation about z-axis by angle (radians)."""
    cosa, sina = np.cos(angle), np.sin(angle)
    return np.array([
        [cosa, -sina, 0],
        [sina,  cosa, 0],
        [ 0,   0, 1]
    ], dtype=float)

def skew(vec: Vector3) -> Matrix3x3:
    """Return the skew-symmetric matrix of a 3D vector."""
    x, y, z = vec.reshape(3,)
    return np.array([
        [ 0, -z,  y],
        [ z,  0, -x],
        [-y,  x,  0]
    ], dtype=float)

def exp_so3(w: Vector3, angle: float) -> Matrix3x3:
    """
    Return rotation matrix ∈ SO(3) for rotation about given axis by angle (radians). 
    Known as Rodrigues' rotation formula.
    """
    norm = np.linalg.norm(w)
    if norm < 1e-8: # TODO: Global constant?
        return np.eye(3, dtype=float)
    w = w / norm
    K = skew(w)
    I = np.eye(3, dtype=float)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def log_so3(R: Matrix3x3) -> Vector3:
    """
    Return the axis of rotation (unit vector) for a given rotation matrix ∈ SO(3).
    The angle of rotation can be computed separately if needed.
    """
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    eps = 1e-8 # TODO; Global constant?

    if angle < eps:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    if np.pi - angle < eps:
        if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
            x = np.sqrt(max(0.0, (R[0, 0] + 1) / 2))
            y = R[0, 1] / (2 * x) if x > eps else 0.0
            z = R[0, 2] / (2 * x) if x > eps else 0.0
        elif R[1, 1] >= R[2, 2]:
            y = np.sqrt(max(0.0, (R[1, 1] + 1) / 2))
            x = R[0, 1] / (2 * y) if y > eps else 0.0
            z = R[1, 2] / (2 * y) if y > eps else 0.0
        else:
            z = np.sqrt(max(0.0, (R[2, 2] + 1) / 2))
            x = R[0, 2] / (2 * z) if z > eps else 0.0
            y = R[1, 2] / (2 * z) if z > eps else 0.0
        axis = np.array([x, y, z], dtype=float)
        norm = np.linalg.norm(axis)
        if norm < eps:
            return np.array([0.0, 0.0, 0.0], dtype=float)
        return axis / norm

    wx = (R - R.T) / (2 * np.sin(angle))
    return np.array([wx[2, 1], wx[0, 2], wx[1, 0]], dtype=float)
