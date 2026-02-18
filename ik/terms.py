from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from core.so3 import log_so3_vec
from ik.types import PoseTarget, IKTolerances, Vec6


class ResidualTerm:
    def residual(self, q: Vec6) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class PoseTrackingTerm(ResidualTerm):
    fk: callable                 # fk(q)->4x4
    target: PoseTarget
    tol: IKTolerances

    def residual(self, q: Vec6) -> np.ndarray:
        T = self.fk(q)
        p = T[:3, 3]
        R = T[:3, :3]

        e_p = (p - self.target.position_mm) / self.tol.pos_mm

        # rotation vector (rad), normalized by tolerance
        e_R = log_so3_vec(self.target.rotation.T @ R) / self.tol.rot_rad

        return np.hstack([e_p, e_R])


@dataclass(frozen=True)
class PostureTerm(ResidualTerm):
    q_pref: np.ndarray
    sigma: np.ndarray
    weight: float = 1.0

    def residual(self, q: Vec6) -> np.ndarray:
        return np.sqrt(self.weight) * ((q - self.q_pref) / self.sigma)


@dataclass(frozen=True)
class SmoothnessTerm(ResidualTerm):
    q_prev: np.ndarray
    sigma: np.ndarray
    weight: float = 1.0

    def residual(self, q: Vec6) -> np.ndarray:
        return np.sqrt(self.weight) * ((q - self.q_prev) / self.sigma)
