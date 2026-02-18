# ik/provider.py
from __future__ import annotations
from typing import Optional, Protocol, Tuple
import numpy as np

class KinematicsProvider(Protocol):
    dof: int

    def fk_end_effector(self, q: np.ndarray) -> np.ndarray:
        """Return 4x4 world_T_ee."""
        ...

    def joint_limits(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (lb, ub) or None."""
        ...
