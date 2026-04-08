import numpy as np

from core.so3 import skew
from core.types import Matrix3x3, Matrix6x6, Vector3


def spatial_inertia(m: float, com: Vector3, I_origin: Matrix3x3) -> Matrix6x6:
    """
    Return the 6x6 spatial inertia matrix using inertia about the frame origin. 
    Units in m and com must match the ones in I_origin.
    """
    C = skew(com)
    return np.block([
        [I_origin, m * C],
        [-m * C, m * np.eye(3)],
    ])
