import numpy as np

from robot.robot import UR5
from robot.viewer import UR5Viewer


def rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


if __name__ == "__main__":
    robot = UR5()
    viewer = UR5Viewer(robot=robot, window_name="UR5 Visual Test")

    home_frames = robot.default_frames()

    def update(t: float) -> None:
        world_spin = rotation_z(0.5 * t)
        transforms = {
            name: world_spin @ home_frame
            for name, home_frame in home_frames.items()
        }
        viewer.set_link_frames(transforms)

    viewer.run(update_callback=update, fps=60)
