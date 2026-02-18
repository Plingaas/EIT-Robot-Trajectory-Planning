from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


Vec3 = np.ndarray  # shape (3,)
Vec6 = np.ndarray  # shape (6,)
Mat3 = np.ndarray  # shape (3,3)
Mat4 = np.ndarray  # shape (4,4)


@dataclass(frozen=True)
class PoseTarget:
    """End-effector target pose in the WORLD frame."""
    position_mm: Vec3           # (x,y,z) in mm
    rotation: Mat3              # 3x3 rotation matrix


@dataclass(frozen=True)
class IKTolerances:
    """Desired accuracy. Used to normalize residuals."""
    pos_mm: float = 0.1
    rot_deg: float = 0.1

    @property
    def rot_rad(self) -> float:
        return float(np.deg2rad(self.rot_deg))


@dataclass(frozen=True)
class SolveOptions:
    method_unbounded: str = "lm"     # LM when no bounds
    method_bounded: str = "trf"      # TRF when bounds exist
    max_nfev: int = 120
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    verbose: int = 0                # 0/1/2 per SciPy
    loss: str = "linear"            # can be "soft_l1" etc if you like


@dataclass
class IKResult:
    q: Vec6
    success: bool
    cost: float
    message: str
    nfev: int
    residual_norm: float
    pose_pos_err_mm: float
    pose_rot_err_deg: float
