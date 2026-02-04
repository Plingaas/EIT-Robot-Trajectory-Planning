import numpy as np
from core.types import Matrix3x3

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