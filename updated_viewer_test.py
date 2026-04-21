import numpy as np

from core.kinematics.fk import fk_all, fk_links
# from robot.ur5e_parameters import M_LIST, S
# from visual.ur5e_model import LINK_ORDER, UR5e
from robot.ur10_parameters import M_LIST, S
from visual.ur10_model import LINK_ORDER, UR10
from visual.viewer import FrameSequence, Viewer

MESHES_ARE_GLOBAL_CAD_EXPORTS = True


def shortest_angular_delta(q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
    return (q_goal - q_start + np.pi) % (2.0 * np.pi) - np.pi


def interpolate_joint_path(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    num_steps: int,
) -> list[np.ndarray]:
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")

    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    dq = shortest_angular_delta(q_start, q_goal)

    samples = []
    for s in np.linspace(0.0, 1.0, num_steps):
        alpha = 0.5 - 0.5 * np.cos(np.pi * s)
        samples.append(q_start + alpha * dq)
    return samples


def build_frame_sequence(
    home_frame_list: list[np.ndarray],
    q_path: list[np.ndarray],
    dt: float,
    meshes_are_global_cad_exports: bool = False,
) -> FrameSequence:
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")

    frames = []
    for q in q_path:
        if meshes_are_global_cad_exports:
            # Global-frame STLs already contain their home offsets, so move
            # meshes by the joint product A_i only.
            motion_frames, _ = fk_all(home_frame_list[0], S, q)
            frame = dict(zip(LINK_ORDER, motion_frames))
            frame["tcp"] = motion_frames[-1] @ home_frame_list[-1]
        else:
            # Link-local STLs need the full link frame A_i @ M_i.
            link_frames = fk_links(home_frame_list, S, q)
            frame = dict(zip(LINK_ORDER, link_frames))
            frame["tcp"] = link_frames[-1]
        frames.append(frame)
    times = (np.arange(len(frames), dtype=float) * dt).tolist()
    return FrameSequence(frames=frames, times=times)


def build_dance_path(poses: list[np.ndarray], steps_per_move: int) -> list[np.ndarray]:
    if len(poses) < 2:
        raise ValueError("poses must contain at least two dance poses.")

    q_path: list[np.ndarray] = []
    for i in range(len(poses)):
        q_start = poses[i]
        q_goal = poses[(i + 1) % len(poses)]
        segment = interpolate_joint_path(q_start, q_goal, num_steps=steps_per_move)
        if q_path:
            segment = segment[1:]
        q_path.extend(segment)
    return q_path


def build_sway_path(num_cycles: int, steps_per_cycle: int) -> list[np.ndarray]:
    q_center = np.deg2rad(np.array([0.0, -75.0, 110.0, -35.0, 90.0, 0.0], dtype=float))
    q_amp = np.deg2rad(np.array([40.0, 22.0, 18.0, 30.0, 20.0, 45.0], dtype=float))

    q_path: list[np.ndarray] = []
    total_steps = num_cycles * steps_per_cycle
    for step in range(total_steps):
        phase = 2.0 * np.pi * step / steps_per_cycle
        sway = np.sin(phase)
        accent = np.sin(2.0 * phase)

        q = q_center.copy()
        q[0] += q_amp[0] * sway
        q[1] += q_amp[1] * sway
        q[2] -= q_amp[2] * sway
        q[3] += q_amp[3] * sway
        q[4] += q_amp[4] * accent
        q[5] += q_amp[5] * sway
        q_path.append(q)

    return q_path


if __name__ == "__main__":
    robot = UR10()
    viewer = Viewer(window_name="UR10 Motion Test", world_frame_size=0.25)
    viewer.add_robot(robot, frame_size=0.1)
    viewer.add_trace("tcp")

    q_path = build_sway_path(num_cycles=6, steps_per_cycle=90)
    sequence = build_frame_sequence(
        M_LIST,
        q_path,
        dt=0.1,
        meshes_are_global_cad_exports=MESHES_ARE_GLOBAL_CAD_EXPORTS,
    )

    viewer.run_sequence(sequence=sequence, loop=True)
