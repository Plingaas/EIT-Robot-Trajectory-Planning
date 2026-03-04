import numpy as np

from core.kinematics.fk import fk_links
from robot.ur5e_parameters import S
from robot.ur5e_home_poses import LINK_ORDER
from visual.ur5e_model import UR5e
from visual.viewer import FrameSequence, Viewer


def interpolate_joint_path(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    num_steps: int,
) -> list[np.ndarray]:
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")

    samples = []
    for s in np.linspace(0.0, 1.0, num_steps):
        # Smooth start/stop profile.
        alpha = 0.5 - 0.5 * np.cos(np.pi * s)
        samples.append((1.0 - alpha) * q_start + alpha * q_goal)
    return samples


def build_frame_sequence(
    home_frame_list: list[np.ndarray],
    q_path: list[np.ndarray],
    fps: float,
) -> FrameSequence:
    frames = []
    for q in q_path:
        link_frames = fk_links(home_frame_list, S, q)
        frames.append(dict(zip(LINK_ORDER, link_frames)))
    return FrameSequence(frames=frames, fps=fps, loop=False)


if __name__ == "__main__":
    robot = UR5e()
    viewer = Viewer(window_name="UR5e Motion Test")
    viewer.add_robot(robot)
    home_frames = robot.home_frames()
    home_frame_list = [home_frames[name] for name in LINK_ORDER]

    q_home = np.zeros(6, dtype=float)
    half = np.pi / 2
    q_goal = np.array([half, -half, half, -half, half, -half], dtype=float)

    q_path = interpolate_joint_path(q_home, q_goal, num_steps=180)
    sequence = build_frame_sequence(home_frame_list, q_path, fps=60.0)

    viewer.run(sequence=sequence)
