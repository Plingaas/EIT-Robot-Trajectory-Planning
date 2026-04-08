import json
from multiprocessing import Process
from pathlib import Path

import numpy as np

from core.kinematics.fk import fk_links
from robot.ur5e_parameters import M_LIST, S
from visual.ur5e_model import LINK_ORDER, UR5e
from visual.viewer import FrameSequence, Viewer


CUBIC_TRAJECTORY_PATH = Path("trajectories/cubic_trajectory.json")
OPTIMIZED_TRAJECTORY_PATH = Path("trajectories/optimized_trajectory.json")
REPLAY_LOOP = True


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
    home_frame_list: tuple[np.ndarray, ...] | list[np.ndarray] = M_LIST,
) -> FrameSequence:
    times, q_path = load_joint_trajectory(path)

    frames = []
    for q in q_path:
        link_frames = fk_links(home_frame_list, S, q)
        frames.append(dict(zip(LINK_ORDER, link_frames)))

    return FrameSequence(frames=frames, times=times.tolist())


def replay_saved_trajectory(
    path: str | Path,
    window_name: str,
    loop: bool = True,
) -> None:
    robot = UR5e()
    viewer = Viewer(window_name=window_name, world_frame_size=0.25)
    viewer.add_robot(robot, frame_size=0.1)
    viewer.add_trace("end_effector")

    sequence = trajectory_file_to_frame_sequence(path)
    viewer.run_sequence(sequence=sequence, loop=loop)


def launch_parallel_replays(
    cubic_path: str | Path,
    optimized_path: str | Path,
    loop: bool = True,
) -> None:
    processes = [
        Process(
            target=replay_saved_trajectory,
            args=(Path(cubic_path), "Replay: Cubic Trajectory", loop),
        ),
        Process(
            target=replay_saved_trajectory,
            args=(Path(optimized_path), "Replay: Optimized Trajectory", loop),
        ),
    ]

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()


if __name__ == "__main__":
    launch_parallel_replays(
        CUBIC_TRAJECTORY_PATH,
        OPTIMIZED_TRAJECTORY_PATH,
        loop=REPLAY_LOOP,
    )
