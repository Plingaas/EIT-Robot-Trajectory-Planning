import numpy as np
from core.types import Matrix4x4, Vector6
from core.se3 import inv_se3, log_se3, vee

def pose_error_space(T: Matrix4x4, T_des: Matrix4x4) -> Vector6:
    """
    Space-frame pose error:

        e_s = vee( log( T_des @ inv(T) ) )

    Args:
        T:     (4,4) current end-effector pose in SE(3)
        T_des: (4,4) desired end-effector pose in SE(3)

    Returns:
        e_s: (6,) error twist expressed in the space frame.
    """
    T = np.asarray(T, dtype=float)
    T_des = np.asarray(T_des, dtype=float)

    if T.shape != (4, 4) or T_des.shape != (4, 4):
        raise ValueError("T and T_des must be 4x4 matrices.")

    T_err = T_des @ inv_se3(T)
    Xi_hat = log_se3(T_err)
    error_twist = vee(Xi_hat)
    return error_twist

