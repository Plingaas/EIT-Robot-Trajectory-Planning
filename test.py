import numpy as np

from core.se3 import exp_se3_twist
from visual.model import UR5
from robot.ur5e_parameters import S
from visual.ur5e_viewer import UR5Viewer


LINK_JOINT_DEPTH = {
    "mount": 0,
    "base": 1,
    "shoulder": 2,
    "elbow": 3,
    "forearm": 4,
    "wrist": 5,
    "end_effector": 6,
}

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


def link_frames_world(q: np.ndarray, home_frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    transforms: dict[str, np.ndarray] = {}
    cumulative = np.eye(4, dtype=float)

    for link_name, depth in LINK_JOINT_DEPTH.items():
        if depth == 0:
            transforms[link_name] = np.array(home_frames[link_name], dtype=float, copy=True)
            continue

        cumulative = cumulative @ exp_se3_twist(S[:, depth - 1], float(q[depth - 1]))
        transforms[link_name] = cumulative @ home_frames[link_name]

    return transforms


if __name__ == "__main__":
    robot = UR5()
    viewer = UR5Viewer(robot=robot, window_name="UR5 Motion Test")
    home_frames = robot.default_frames()

    def update(t: float) -> None:
        q = sample_joint_angles(t)
        viewer.set_link_frames(link_frames_world(q, home_frames))

    viewer.run(update_callback=update, fps=60)
