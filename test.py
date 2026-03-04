import numpy as np

from core.kinematics.fk import fk_links
from robot.ur5e_parameters import S
from robot.ur5e_home_poses import LINK_ORDER
from robot.ur5e_robot import UR5e
from visual.viewer import Viewer

def sample_joint_angles(t: float) -> np.ndarray:
    return np.array(
        [
            0.35 * np.sin(0.5 * t),
            0.45 * np.sin(0.8 * t),
            -0.35 * np.sin(0.8 * t + 0.6),
            0.30 * np.sin(1.1 * t + 0.4),
            0.40 * np.sin(0.6 * t + 0.2),
            0.55 * np.sin(1.4 * t),
        ],
        dtype=float,
    )


if __name__ == "__main__":
    robot = UR5e()
    viewer = Viewer(window_name="UR5e Motion Test")
    viewer.add_robot(robot)
    home_frames = robot.home_frames()
    home_frame_list = [home_frames[name] for name in LINK_ORDER]

    def update(t: float) -> None:
        q = sample_joint_angles(t)
        link_frames = fk_links(home_frame_list, S, q)
        transforms = dict(zip(LINK_ORDER, link_frames))
        viewer.set_transforms(transforms)

    viewer.run(update_callback=update, fps=60)
