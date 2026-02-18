# ik/providers/ur5.py
from __future__ import annotations
import numpy as np

from ik.provider import KinematicsProvider
from robot.robot import UR5  # your class

class UR5KinematicsProvider:
    dof: int = 6

    def __init__(self, robot: UR5):
        self._robot = robot

    def fk_end_effector(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(6,)
        return self._robot.fk_end_effector(q)

    def joint_limits(self):
        return self._robot.joint_limits()
