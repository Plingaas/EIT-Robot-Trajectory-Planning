from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .types import Mat4, Vec6


class RobotModel(ABC):
    """Minimal interface the IK solver needs."""

    @property
    @abstractmethod
    def dof(self) -> int: ...

    @abstractmethod
    def fk_end_effector(self, q: Vec6) -> Mat4:
        """Returns world_T_ee for configuration q."""
        ...

    def joint_limits(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Optional (lb, ub). Override if you have limits."""
        return None


@dataclass(frozen=True)
class JointPosture:
    """
    Soft preference for 'natural' configurations.

    q_pref: preferred joints (len=dof)
    sigma: per-joint "allowed deviation" (rad). Smaller => stronger.
    weight: overall multiplier (dimensionless).
    """
    q_pref: np.ndarray
    sigma: np.ndarray
    weight: float = 1.0


@dataclass(frozen=True)
class Smoothness:
    """
    Soft preference to stay near previous q (branch-locking).
    """
    q_prev: np.ndarray
    sigma: np.ndarray
    weight: float = 1.0
