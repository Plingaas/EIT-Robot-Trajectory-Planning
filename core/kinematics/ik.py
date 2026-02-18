import numpy as np
from core.types import Matrix4x4, Matrix6xn, Vectorn
from core.kinematics.fk import fk
from core.kinematics.jacobian import jacobian_space
from core.kinematics.error import pose_error_space
from core.kinematics.validation import validate_poe_inputs, validate_homogeneous_transform

# TODO; Consider weighting of position vs orientation error in the DLS step. 

def _damped_least_squares(J: np.ndarray, err: np.ndarray, damping: float) -> np.ndarray:
    """
    Compute the damped least squares solution to J dq = err.

    Args:
        J: (6,n) Jacobian matrix
        err: (6,) error twist
        damping: scalar damping factor

    Returns:
        dq: (n,) joint angle update
    """
    I6 = np.eye(6)
    JJt = J @ J.T                      
    dq = J.T @ np.linalg.solve(JJt + (damping**2) * I6, err)  
    return dq

def ik_space_dls(
    M: Matrix4x4,
    S: Matrix6xn,
    q0: Vectorn,
    T_des: Matrix4x4,
    max_iterations: int = 100,
    tolerance: float = 0.0001,
    damping: float = 0.01,
    step_size: float = 1.0,
    joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, bool, dict]:
    """
    Numerical IK in space frame using damped least squares.

    Returns:
        q: (n,) solution (best found)
        success: bool
        info: dict with diagnostics (iters, final_error_norm)
    """
    M = np.asarray(M, dtype=float)
    S = np.asarray(S, dtype=float)
    q = np.asarray(q0, dtype=float).reshape(-1)
    T_des = np.asarray(T_des, dtype=float)

    validate_poe_inputs(M, S, q)
    validate_homogeneous_transform(T_des)
    
    success = False
    for iteration in range(max_iterations):
        T_now = fk(M, S, q)
        err = pose_error_space(T_now, T_des)     
        err_norm = np.linalg.norm(err)

        if err_norm < tolerance:
            success = True
            break

        J = jacobian_space(M, S, q)    
        dq = _damped_least_squares(J, err, damping)

        q = q + step_size * dq

        if joint_limits is not None:
            q_min, q_max = joint_limits
            q = np.clip(q, q_min, q_max)
            
    return q, success, {"iterations": iteration, "final_error_norm": err_norm}