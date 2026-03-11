import json
from pathlib import Path

import numpy as np

from core.kinematics.fk import fk_links
from robot.ur5e_home_poses import LINK_ORDER
from robot.ur5e_parameters import S
from visual.ur5e_model import UR5e
from visual.viewer import FrameSequence, Viewer


def load_trajectory(path: str | Path) -> tuple[list[float], list[np.ndarray]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    units = data.get("units", "deg").lower()
    if units not in {"deg", "rad"}:
        raise ValueError(f"Unsupported trajectory units '{units}'. Expected 'deg' or 'rad'.")

    times: list[float] = []
    joint_positions: list[np.ndarray] = []

    for waypoint in sorted(data["waypoints"], key=lambda item: float(item["t"])):
        t = float(waypoint["t"])
        q = np.asarray(waypoint["q"], dtype=float)
        if q.shape != (6,):
            raise ValueError(f"Waypoint q must have 6 values, got shape {q.shape}")
        if units == "deg":
            q = np.deg2rad(q)

        times.append(t)
        joint_positions.append(q)

    if len(times) < 2:
        raise ValueError("Need at least 2 waypoints.")
    if times[0] < 0.0:
        raise ValueError("Waypoint times must be >= 0.")
    if any(t1 <= t0 for t0, t1 in zip(times, times[1:])):
        raise ValueError("Waypoint times must be strictly increasing.")

    return times, joint_positions


def build_frame_sequence(
    times: list[float],
    joint_positions: list[np.ndarray],
    home_frame_list: list[np.ndarray],
) -> FrameSequence:
    frames = []
    for q in joint_positions:
        link_frames = fk_links(home_frame_list, S, q)
        frames.append(dict(zip(LINK_ORDER, link_frames)))
    return FrameSequence(frames=frames, times=times)


def main() -> None:
    robot = UR5e()
    viewer = Viewer(window_name="UR5 Trajectory Player", world_frame_size=300.0)
    viewer.add_robot(robot, show_frames=False, frame_size=100)
    viewer.add_trace("end_effector")

    home_frames = robot.home_frames()
    home_frame_list = [home_frames[name] for name in LINK_ORDER]
    times, joint_positions = load_trajectory("trajectory.json")

    sequence = build_frame_sequence(times, joint_positions, home_frame_list)
    viewer.run_sequence(sequence, loop=True)


if __name__ == "__main__":
    main()
