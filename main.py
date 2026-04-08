import json
from pathlib import Path

import numpy as np

from core.kinematics.fk import fk_links
from robot.ur5e_parameters import M_LIST, S
from visual.ur5e_model import LINK_ORDER, UR5e
from visual.viewer import Viewer


def load_trajectory(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    units = data.get("units", "deg").lower()
    if units not in {"deg", "rad"}:
        raise ValueError(f"Unsupported trajectory units '{units}'. Expected 'deg' or 'rad'.")

    waypoints = sorted(data["waypoints"], key=lambda item: float(item["t"]))

    times = []
    joint_positions = []

    for waypoint in waypoints:
        t = float(waypoint["t"])
        q = np.asarray(waypoint["q"], dtype=float)
        if q.shape != (6,):
            raise ValueError(f"Waypoint q must have 6 values, got shape {q.shape}")
        if units == "deg":
            q = np.deg2rad(q)

        times.append(t)
        joint_positions.append(q)

    times = np.asarray(times, dtype=float)
    joint_positions = np.asarray(joint_positions, dtype=float)

    if len(times) < 2:
        raise ValueError("Need at least 2 waypoints.")
    if times[0] < 0.0:
        raise ValueError("Waypoint times must be >= 0.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Waypoint times must be strictly increasing.")

    return times, joint_positions


def interpolate_q(t: float, times: np.ndarray, q_waypoints: np.ndarray) -> np.ndarray:
    if t <= times[0]:
        return q_waypoints[0]
    if t >= times[-1]:
        return q_waypoints[-1]

    i = np.searchsorted(times, t) - 1
    t0 = times[i]
    t1 = times[i + 1]
    q0 = q_waypoints[i]
    q1 = q_waypoints[i + 1]

    alpha = (t - t0) / (t1 - t0)
    return (1.0 - alpha) * q0 + alpha * q1


def main() -> None:
    robot = UR5e()
    viewer = Viewer(window_name="UR5 Trajectory Player", world_frame_size=300.0)
    viewer.add_robot(robot, show_frames=False, frame_size=100.0)
    viewer.add_trace("end_effector")

    times, joint_positions = load_trajectory("trajectory.json")
    duration = float(times[-1])

    def update(t: float) -> None:
        t_wrapped = t % duration
        q = interpolate_q(t_wrapped, times, joint_positions)
        link_frames = fk_links(M_LIST, S, q)
        viewer.set_transforms(dict(zip(LINK_ORDER, link_frames)))

    viewer.run_callback(update, fps=60.0)


if __name__ == "__main__":
    main()
