import numpy as np
from core.types import Matrix3x3, Matrix4x4, Vector3

def homogeneous(R: Matrix3x3, p: Vector3) -> Matrix4x4:
    """Build 4x4 homogeneous transform ∈ SE(3) from R ∈ SO(3) and p ∈ ℝ³"""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p.reshape(3,)
    return T

def translate(x: float, y: float, z: float) -> Matrix4x4:
    """Return homogeneous transform ∈ SE(3) for translation by (x, y, z)."""
    return homogeneous(np.eye(3), np.array([x, y, z], dtype=float))