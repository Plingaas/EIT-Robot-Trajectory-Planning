import json
from pathlib import Path

import numpy as np

from core.kinematics.fk import fk_links
from robot.ur5e_home_poses import LINK_ORDER
from robot.ur5e_parameters import S
from visual.ur5e_model import UR5e
from visual.viewer import FrameSequence, Viewer


def load_trajectory(path: str | Path) -> tuple[str, list[tuple[float, np.ndarray]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    units = data.get("units", "deg").lower()
    if units not in {"deg", "rad"}:
        raise ValueError(f"Unsupported trajectory units '{units}'. Expected 'deg' or 'rad'.")

    waypoints = []
    for waypoint in sorted(data["waypoints"], key=lambda item: float(item["t"])):
        t = float(waypoint["t"])
        q = np.asarray(waypoint["q"], dtype=float)
        if q.shape != (6,):
            raise ValueError(f"Waypoint q must have 6 values, got shape {q.shape}")
        waypoints.append((t, q))

    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints.")
    if waypoints[0][0] < 0.0:
        raise ValueError("Waypoint times must be >= 0.")
    if waypoints[-1][0] <= 0.0:
        raise ValueError("Last waypoint time must be > 0.")

    return units, waypoints


def sample_trajectory(waypoints: list[tuple[float, np.ndarray]], t: float) -> np.ndarray:
    duration = waypoints[-1][0]
    t = t % duration

    for i in range(len(waypoints) - 1):
        t0, q0 = waypoints[i]
        t1, q1 = waypoints[i + 1]
        if t0 <= t < t1:
            alpha = (t - t0) / (t1 - t0)
            return (1.0 - alpha) * q0 + alpha * q1

    return waypoints[0][1].copy()


def build_frame_sequence(
    waypoints: list[tuple[float, np.ndarray]],
    units: str,
    home_frame_list: list[np.ndarray],
    fps: float = 120.0,
    loop: bool = True,
) -> FrameSequence:
    if fps <= 0.0:
        raise ValueError("fps must be > 0.")

    duration = waypoints[-1][0]
    sample_times = np.arange(0.0, duration, 1.0 / fps)
    if sample_times.size == 0:
        sample_times = np.array([0.0], dtype=float)

    frames = []
    for t in sample_times:
        q = sample_trajectory(waypoints, t)
        if units == "deg":
            q = np.deg2rad(q)
        link_frames = fk_links(home_frame_list, S, q)
        frames.append(dict(zip(LINK_ORDER, link_frames)))

    return FrameSequence(frames=frames, fps=fps, loop=loop)


def main() -> None:
    robot = UR5e()
    viewer = Viewer(window_name="UR5 Trajectory Player", world_frame_size=300.0)
    viewer.add_robot(robot, show_frames=True, frame_size=120.0)
    viewer.add_trace("end_effector")

    home_frames = robot.home_frames()
    home_frame_list = [home_frames[name] for name in LINK_ORDER]
    units, waypoints = load_trajectory("trajectory.json")

    sequence = build_frame_sequence(waypoints, units, home_frame_list)
    viewer.run(sequence=sequence)


if __name__ == "__main__":
    main()
