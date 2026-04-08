import numpy as np

from pathlib import Path

from visual.model import VisualLink, VisualRobot
from robot.ur5e_parameters import HOME_FRAMES

ROOT = Path(__file__).resolve().parent.parent
MM_TO_M = 1e-3

LINK_ORDER = (
    "mount",
    "base",
    "shoulder",
    "elbow",
    "forearm",
    "wrist",
    "end_effector",
)

MESH_PATHS = {
    "mount": "model/base_mount_fixed.stl",
    "base": "model/base_joint_fixed.stl",
    "shoulder": "model/shoulder_joint_fixed.stl",
    "elbow": "model/elbow_joint_fixed.stl",
    "forearm": "model/forearm_joint_fixed.stl",
    "wrist": "model/wrist_joint_fixed.stl",
    "end_effector": "model/end_effector_joint_fixed.stl",
}

class UR5e(VisualRobot):
    def __init__(self):
        links = []
        for name in LINK_ORDER:
            links.append(
                VisualLink(
                    name=name,
                    mesh_path=ROOT / MESH_PATHS[name],
                    home_frame=np.array(HOME_FRAMES[name], dtype=float, copy=True),
                    mesh_scale=MM_TO_M,
                )
            )
        super().__init__(links)
