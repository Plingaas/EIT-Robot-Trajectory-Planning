import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.kinematics.fk import fk_links
from robot.ur5e_home_poses import LINK_ORDER
from robot.ur5e_parameters import S
from visual.ur5e_model import UR5e
from visual.viewer import FrameSequence, Viewer


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

    def build_frame_sequence(
        self,
        home_frame_list: list[np.ndarray],
        fps: float = 60.0,
        loop: bool = True,
    ) -> FrameSequence:
        if fps <= 0.0:
            raise ValueError("fps must be > 0.")

        sample_count = max(2, int(np.ceil(self.duration * fps)))
        sample_times = np.arange(sample_count, dtype=float) / fps

        frames = []
        for t in sample_times:
            q = self.sample_radians(t)
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

    traj = TrajectoryPlayer.from_json("trajectory.json")
    print("units:", traj.units)
    print("last waypoint:", traj.waypoints[-1].t, traj.waypoints[-1].q)
    sequence = traj.build_frame_sequence(home_frame_list, fps=60.0, loop=True)
    viewer.run(sequence=sequence)


if __name__ == "__main__":
    main()
