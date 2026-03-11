import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from core.kinematics.fk import fk_links
from robot.ur5e_home_poses import LINK_ORDER
from robot.ur5e_parameters import S
from visual.ur5e_model import UR5e
from visual.viewer import Viewer


class TracePath:
    def __init__(self, viewer: Viewer, color: tuple[float, float, float] = (0.0, 1.0, 0.0)):
        self._viewer = viewer
        self._color = np.asarray(color, dtype=float)
        self._points: list[np.ndarray] = []
        self._line_set = o3d.geometry.LineSet()
        self._line_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self._line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        self._line_set.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self._viewer.vis.add_geometry(self._line_set)

    def add_point(self, point_world: np.ndarray, max_points: int = 5000) -> None:
        point_world = np.asarray(point_world, dtype=float).reshape(3,)
        self._points.append(point_world)
        if len(self._points) > max_points:
            self._points = self._points[-max_points:]

        if len(self._points) < 2:
            return

        points = np.vstack(self._points)
        lines = np.column_stack(
            [np.arange(len(points) - 1), np.arange(1, len(points))]
        ).astype(np.int32)
        colors = np.tile(self._color, (len(lines), 1))

        self._line_set.points = o3d.utility.Vector3dVector(points)
        self._line_set.lines = o3d.utility.Vector2iVector(lines)
        self._line_set.colors = o3d.utility.Vector3dVector(colors)
        self._viewer.vis.update_geometry(self._line_set)


@dataclass(frozen=True)
class Waypoint:
    t: float
    q: np.ndarray


class TrajectoryPlayer:
    def __init__(self, waypoints: list[Waypoint], units: str = "deg"):
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        waypoints = sorted(waypoints, key=lambda waypoint: waypoint.t)
        if any(waypoint.q.shape != (6,) for waypoint in waypoints):
            raise ValueError("Each waypoint q must be length 6")
        if waypoints[0].t < 0:
            raise ValueError("Waypoint times must be >= 0")

        self.waypoints = waypoints
        self.units = units.lower()
        self.duration = waypoints[-1].t
        if self.duration <= 0:
            raise ValueError("Last waypoint time must be > 0")

    @staticmethod
    def from_json(path: str | Path) -> "TrajectoryPlayer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        waypoints: list[Waypoint] = []
        for waypoint in data["waypoints"]:
            t = float(waypoint["t"])
            q = np.array(waypoint["q"], dtype=float)
            if q.shape != (6,):
                raise ValueError(f"Waypoint q must have 6 values, got shape {q.shape}")
            waypoints.append(Waypoint(t=t, q=q))

        return TrajectoryPlayer(waypoints, units=data.get("units", "deg"))

    def sample(self, t: float) -> np.ndarray:
        tt = t % self.duration
        waypoints = self.waypoints

        i = 0
        while i + 1 < len(waypoints) and not (waypoints[i].t <= tt < waypoints[i + 1].t):
            i += 1

        if i + 1 >= len(waypoints):
            i = 0

        t0, q0 = waypoints[i].t, waypoints[i].q
        t1, q1 = waypoints[i + 1].t, waypoints[i + 1].q
        if t1 <= t0:
            return q0.copy()

        s = (tt - t0) / (t1 - t0)
        return (1.0 - s) * q0 + s * q1

    def sample_radians(self, t: float) -> np.ndarray:
        q = self.sample(t)
        if self.units == "deg":
            return np.deg2rad(q)
        if self.units == "rad":
            return q
        raise ValueError(f"Unsupported trajectory units '{self.units}'. Expected 'deg' or 'rad'.")


def main() -> None:
    robot = UR5e()
    viewer = Viewer(window_name="UR5 Trajectory Player", world_frame_size=300.0)
    viewer.add_robot(robot, show_frames=True, frame_size=120.0)

    trace = TracePath(viewer)
    home_frames = robot.home_frames()
    home_frame_list = [home_frames[name] for name in LINK_ORDER]

    traj = TrajectoryPlayer.from_json("trajectory.json")
    print("units:", traj.units)
    print("last waypoint:", traj.waypoints[-1].t, traj.waypoints[-1].q)

    def update(t_now: float) -> None:
        q = traj.sample_radians(t_now)
        link_frames = fk_links(home_frame_list, S, q)
        viewer.set_transforms(dict(zip(LINK_ORDER, link_frames)))
        trace.add_point(link_frames[-1][:3, 3])

    viewer.run(update_callback=update, fps=60.0)


if __name__ == "__main__":
    main()
