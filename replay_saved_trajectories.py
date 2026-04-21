import json
from pathlib import Path

import numpy as np

from core.kinematics.fk import fk_links
from robot.ur5e_parameters import M_LIST, S
from visual.ur5e_model import LINK_ORDER, UR5e
from visual.viewer import FrameSequence, Viewer


QUINTIC_TRAJECTORY_PATH = Path("trajectories/quintic_trajectory.json")
OPTIMIZED_TRAJECTORY_PATH = Path("trajectories/optimized_trajectory.json")
REPLAY_LOOP = True
QUINTIC_OFFSET = np.array([0.0, -0.4, 0.0], dtype=float)
OPTIMIZED_OFFSET = np.array([0.0, 0.4, 0.0], dtype=float)


def load_joint_trajectory(path: str | Path) -> tuple[np.ndarray, list[np.ndarray]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    units = data.get("units", "rad").lower()
    if units not in {"rad", "deg"}:
        raise ValueError(f"Unsupported trajectory units '{units}'. Expected 'rad' or 'deg'.")

    waypoints = sorted(data["waypoints"], key=lambda item: float(item["t"]))
    times = np.asarray([float(item["t"]) for item in waypoints], dtype=float)
    q_path = [np.asarray(item["q"], dtype=float) for item in waypoints]

    if units == "deg":
        q_path = [np.deg2rad(q) for q in q_path]

    if len(q_path) < 2:
        raise ValueError("Saved trajectory must contain at least two waypoints.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Saved trajectory times must be strictly increasing.")

    return times, q_path


def trajectory_file_to_frame_sequence(
    path: str | Path,
    name_prefix: str = "",
    world_offset: np.ndarray | None = None,
    home_frame_list: tuple[np.ndarray, ...] | list[np.ndarray] = M_LIST,
) -> FrameSequence:
    times, q_path = load_joint_trajectory(path)
    world_offset = np.zeros(3, dtype=float) if world_offset is None else np.asarray(world_offset, dtype=float)

    frames = []
    for q in q_path:
        link_frames = fk_links(home_frame_list, S, q)
        frame = {}
        for name, transform in zip(LINK_ORDER, link_frames):
            T = np.array(transform, dtype=float, copy=True)
            T[:3, 3] += world_offset
            frame[f"{name_prefix}{name}"] = T
        frames.append(frame)

    return FrameSequence(frames=frames, times=times.tolist())


def add_robot_to_viewer(
    viewer: Viewer,
    robot: UR5e,
    name_prefix: str,
    world_offset: np.ndarray,
    trace_color: tuple[float, float, float],
) -> None:
    world_offset = np.asarray(world_offset, dtype=float)
    for link in robot.links:
        T = np.array(link.home_frame, dtype=float, copy=True)
        T[:3, 3] += world_offset
        viewer.add_mesh(
            name=f"{name_prefix}{link.name}",
            mesh=robot.load_mesh(link.name),
            transform=T,
            frame_size=0.1,
        )
    viewer.add_trace(f"{name_prefix}end_effector", color=trace_color)


def merge_frame_sequences(*sequences: FrameSequence) -> FrameSequence:
    if not sequences:
        raise ValueError("At least one FrameSequence is required.")

    reference_times = list(sequences[0].times)
    for sequence in sequences[1:]:
        if len(sequence.times) != len(reference_times):
            raise ValueError("All FrameSequences must have the same number of frames.")
        if not np.allclose(sequence.times, reference_times):
            raise ValueError("All FrameSequences must share the same timestamps.")

    merged_frames = []
    for frame_group in zip(*(sequence.frames for sequence in sequences)):
        merged_frame = {}
        for frame in frame_group:
            merged_frame.update(frame)
        merged_frames.append(merged_frame)

    return FrameSequence(frames=merged_frames, times=reference_times)


def replay_saved_trajectories_in_same_window(
    quintic_path: str | Path,
    optimized_path: str | Path,
    loop: bool = True,
) -> None:
    viewer = Viewer(window_name="Replay: Quintic vs Optimized", world_frame_size=0.25)
    robot = UR5e()

    add_robot_to_viewer(
        viewer=viewer,
        robot=robot,
        name_prefix="quintic_",
        world_offset=QUINTIC_OFFSET,
        trace_color=(0.0, 0.8, 0.0),
    )
    add_robot_to_viewer(
        viewer=viewer,
        robot=robot,
        name_prefix="optimized_",
        world_offset=OPTIMIZED_OFFSET,
        trace_color=(0.9, 0.2, 0.2),
    )

    quintic_sequence = trajectory_file_to_frame_sequence(
        quintic_path,
        name_prefix="quintic_",
        world_offset=QUINTIC_OFFSET,
    )
    optimized_sequence = trajectory_file_to_frame_sequence(
        optimized_path,
        name_prefix="optimized_",
        world_offset=OPTIMIZED_OFFSET,
    )
    merged_sequence = merge_frame_sequences(quintic_sequence, optimized_sequence)
    viewer.run_sequence(merged_sequence, loop=loop)


if __name__ == "__main__":
    replay_saved_trajectories_in_same_window(
        QUINTIC_TRAJECTORY_PATH,
        OPTIMIZED_TRAJECTORY_PATH,
        loop=REPLAY_LOOP,
    )
