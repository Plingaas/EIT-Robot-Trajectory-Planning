import numpy as np

from pathlib import Path

from robot.robot import VisualLink, VisualRobot
from robot.ur5e_home_poses import HOME_FRAMES, MESH_PATHS, LINK_ORDER

ROOT = Path(__file__).resolve().parent.parent

class UR5e(VisualRobot):
    def __init__(self):
        links = []
        for name in LINK_ORDER:
            links.append(
                VisualLink(
                    name=name,
                    mesh_path=ROOT / MESH_PATHS[name],
                    home_frame=np.array(HOME_FRAMES[name], dtype=float, copy=True),
                )
            )
        super().__init__(links)
