import numpy as np

from pathlib import Path

from visual.model import VisualLink, VisualRobot
from robot.ur10_parameters import STL_HOME_FRAMES

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
    "mount": "ur10_stl/mount.stl",
    "base": "ur10_stl/base.stl",
    "shoulder": "ur10_stl/shoulder.stl",
    "elbow": "ur10_stl/elbow.stl",
    "forearm": "ur10_stl/forearm.stl",
    "wrist": "ur10_stl/wrist.stl",
    "end_effector": "ur10_stl/end_effector.stl",
}

class UR10(VisualRobot):
    def __init__(self):
        links = []
        for name in LINK_ORDER:
            links.append(
                VisualLink(
                    name=name,
                    mesh_path=ROOT / MESH_PATHS[name],
                    home_frame=np.array(STL_HOME_FRAMES[name], dtype=float, copy=True),
                    mesh_scale=MM_TO_M,
                )
            )
        super().__init__(links)
