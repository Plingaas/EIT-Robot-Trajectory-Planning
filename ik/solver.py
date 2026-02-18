from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from ik.provider import KinematicsProvider
from core.so3 import log_so3_vec   # add log_so3_vec as discussed
from utils.transformation_helper import rotate  # or wherever your rotate lives


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    cost: float
    message: str


def solve_ik(
    provider: KinematicsProvider,
    xyz_mm: np.ndarray,
    rxyz: np.ndarray,
    q0: np.ndarray,
    *,
    q_prev: np.ndarray | None = None,
    q_pref: np.ndarray | None = None,
    pos_tol_mm: float = 0.1,
    rot_tol_deg: float = 0.1,
    posture_weight: float = 0.5,
    smooth_weight: float = 0.2,
    posture_sigma: np.ndarray | None = None,
    smooth_sigma: np.ndarray | None = None,
    max_nfev: int = 120,
) -> IKResult:
    """
    LM/TRF IK that depends only on the provider.
    - xyz_mm in mm
    - rxyz in radians (must match your rotate convention)
    """
    xyz_mm = np.asarray(xyz_mm, float).reshape(3,)
    rxyz = np.asarray(rxyz, float).reshape(3,)
    q0 = np.asarray(q0, float).reshape(provider.dof,)

    rot_tol_rad = float(np.deg2rad(rot_tol_deg))
    R_target = rotate(rxyz[0], rxyz[1], rxyz[2])  # MUST match project convention

    # Default sigmas: only strongly influence shoulder+elbow
    if posture_sigma is None:
        posture_sigma = np.array([999.0, 0.25, 0.25, 999.0, 999.0, 999.0], dtype=float)
    if smooth_sigma is None:
        smooth_sigma = np.array([0.8, 0.25, 0.25, 1.0, 1.0, 1.0], dtype=float)

    def residual(q: np.ndarray) -> np.ndarray:
        T = provider.fk_end_effector(q)
        p = T[:3, 3]
        R = T[:3, :3]

        e_p = (p - xyz_mm) / pos_tol_mm
        e_R = log_so3_vec(R_target.T @ R) / rot_tol_rad  # rotation vector normalized

        r = [e_p, e_R]

        if q_pref is not None:
            r_posture = np.sqrt(posture_weight) * ((q - q_pref) / posture_sigma)
            r.append(r_posture)

        if q_prev is not None:
            r_smooth = np.sqrt(smooth_weight) * ((q - q_prev) / smooth_sigma)
            r.append(r_smooth)

        return np.hstack(r)

    bounds = provider.joint_limits()
    if bounds is None:
        res = least_squares(residual, q0, method="lm", max_nfev=max_nfev)
    else:
        lb, ub = bounds
        res = least_squares(residual, q0, method="trf", bounds=(lb, ub), max_nfev=max_nfev)

    return IKResult(q=res.x, success=bool(res.success), cost=float(res.cost), message=str(res.message))
