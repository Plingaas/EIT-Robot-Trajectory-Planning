from __future__ import annotations

import numpy as np
from typing import Optional

from .model import RobotModel
from .types import Mat4, Vec6


# Ideally import these from your existing math utilities so conventions match 1:1.
# Replace these imports with your real ones.
def rotate(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx

def translate(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = [x, y, z]
    return T

def assemble_T(R: np.ndarray, T_trans: np.ndarray) -> np.ndarray:
    T = np.array(T_trans, dtype=float, copy=True)
    T[:3, :3] = R
    return T


class UR5ChainModel(RobotModel):
    @property
    def dof(self) -> int:
        return 6

    def joint_limits(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        # Put real limits here if you have them. Otherwise return None.
        # Example: +/- 360deg:
        lim = 2.0 * np.pi
        lb = np.full(6, -lim, dtype=float)
        ub = np.full(6, +lim, dtype=float)
        return (lb, ub)

    def fk_end_effector(self, q: Vec6) -> Mat4:
        q1, q2, q3, q4, q5, q6 = q

        R_base_world = rotate(0, 0, q1)
        t_base_world = translate(0, 0, 162.5)
        T_base_world = assemble_T(R_base_world, t_base_world)

        R_shoulder_base = rotate(0, q2, 0)
        t_shoulder_base = translate(0, -137.8, 0)
        T_shoulder_base = assemble_T(R_shoulder_base, t_shoulder_base)
        T_shoulder_world = T_base_world @ T_shoulder_base

        R_elbow_shoulder = rotate(0, q3, 0)
        t_elbow_shoulder = translate(0, 131.8, 425)
        T_elbow_shoulder = assemble_T(R_elbow_shoulder, t_elbow_shoulder)
        T_elbow_world = T_shoulder_world @ T_elbow_shoulder

        R_forearm_elbow = rotate(0, q4, 0)
        t_forearm_elbow = translate(0, -126.7 - 0.6, 392.2)
        T_forearm_elbow = assemble_T(R_forearm_elbow, t_forearm_elbow)
        T_forearm_world = T_elbow_world @ T_forearm_elbow

        R_wrist_forearm = rotate(0, 0, q5 - np.pi / 2)
        t_wrist_forearm = translate(0, 0, 99.7)
        T_wrist_forearm = assemble_T(R_wrist_forearm, t_wrist_forearm)
        T_wrist_world = T_forearm_world @ T_wrist_forearm

        R_end_effector_wrist = rotate(q6, 0, 0)
        t_end_effector_wrist = translate(98.9 + 0.7, 0, 0)
        T_end_effector_wrist = assemble_T(R_end_effector_wrist, t_end_effector_wrist)

        return T_wrist_world @ T_end_effector_wrist
