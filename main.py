import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from robot.robot import UR5
from core.types import Vector6

class UR5Viewer:
    def __init__(self, robot: UR5, world_frame_size: float = 300.0):
        self.robot = robot
        self.vis = o3d.visualization.Visualizer()

        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=world_frame_size)

        # IMPORTANT: these must be stable objects (not recreated every call)
        self._meshes = self.robot.meshes()
        self._frames = self.robot.frames()

        self._trace_pts: list[np.ndarray] = []
        self._trace = o3d.geometry.LineSet()
        self._trace.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self._trace.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        self._trace.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        self._geoms = [self.world_frame] + self._meshes + self._frames + [self._trace]

    def open(self, title="UR5 Viewer", width=1280, height=800):
        self.vis.create_window(title, width=width, height=height)
        for g in self._geoms:
            self.vis.add_geometry(g)

    def tick(self):
        for g in (self._meshes + self._frames + [self._trace]):
            self.vis.update_geometry(g)
        self.vis.poll_events()
        self.vis.update_renderer()

    def is_open(self) -> bool:
        return self.vis.poll_events()

    def close(self):
        self.vis.destroy_window()

    def add_trace_point(self, p_world: np.ndarray, max_points: int = 5000):
        p_world = np.asarray(p_world, dtype=float).reshape(3,)
        self._trace_pts.append(p_world)
    
        if len(self._trace_pts) > max_points:
            self._trace_pts = self._trace_pts[-max_points:]
    
        if len(self._trace_pts) < 2:
            return
    
        P = np.vstack(self._trace_pts)
        L = np.column_stack([np.arange(len(P) - 1), np.arange(1, len(P))]).astype(np.int32)
        C = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(L), 1))
    
        self._trace.points = o3d.utility.Vector3dVector(P)
        self._trace.lines = o3d.utility.Vector2iVector(L)
        self._trace.colors = o3d.utility.Vector3dVector(C)
    
    
@dataclass(frozen=True)
class Waypoint:
    t: float              # seconds
    q: np.ndarray         # shape (6,)


class TrajectoryPlayer:
    def __init__(self, waypoints: list[Waypoint], units: str = "deg"):
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        # sort by time
        waypoints = sorted(waypoints, key=lambda w: w.t)
        if any(w.q.shape != (6,) for w in waypoints):
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

        units = data.get("units", "deg")
        wps_raw = data["waypoints"]

        waypoints = []
        for w in wps_raw:
            t = float(w["t"])
            q = np.array(w["q"], dtype=float)
            if q.shape != (6,):
                raise ValueError(f"Waypoint q must have 6 values, got shape {q.shape}")
            waypoints.append(Waypoint(t=t, q=q))

        return TrajectoryPlayer(waypoints, units=units)

    def sample(self, t: float) -> np.ndarray:
        """
        Sample joint angles at time t (seconds) using linear interpolation.
        Loops automatically: t can be any value.
        """
        # loop time into [0, duration)
        tt = t % self.duration

        # find segment index i such that w[i].t <= tt < w[i+1].t
        w = self.waypoints
        # small lists -> simple linear scan is fine
        i = 0
        while i + 1 < len(w) and not (w[i].t <= tt < w[i + 1].t):
            i += 1

        # edge case: if tt is exactly duration (or past last due to float), wrap to first segment
        if i + 1 >= len(w):
            i = 0

        t0, q0 = w[i].t, w[i].q
        t1, q1 = w[i + 1].t, w[i + 1].q

        if t1 <= t0:
            return q0.copy()

        s = (tt - t0) / (t1 - t0)
        return (1.0 - s) * q0 + s * q1

    def run(self, viewer: UR5Viewer, robot: UR5, hz: float = 60.0):
        dt = 1.0 / hz
        t_start = time.perf_counter()

        while viewer.is_open():
            t_now = time.perf_counter() - t_start
            q = self.sample(t_now)

            # If your robot expects degrees, keep as-is. If it expects radians, convert here.
            # We'll assume degrees if units == "deg".
            if self.units == "deg":
                pose = np.array([*np.deg2rad(q)])  # only if your JointState is degrees-based
            else:
                pose = np.array([*q])

            robot.set_joint_pose(pose)
            viewer.add_trace_point(robot.end_effector_position())
            viewer.tick()
            time.sleep(dt)


def main():
    robot = UR5()
    robot.set_joint_pose(np.array([0, 0, 0, 0, 0, 0]))

    viewer = UR5Viewer(robot, world_frame_size=300.0)
    viewer.open("UR5 Trajectory Player")

    traj = TrajectoryPlayer.from_json("trajectory.json")
    print("units:", traj.units)
    print("last waypoint:", traj.waypoints[-1].t, traj.waypoints[-1].q)
    

    traj.run(viewer, robot, hz=60.0)

    viewer.close()


if __name__ == "__main__":
    main()
