import copy
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import open3d as o3d

from core.kinematics.validation import validate_homogeneous_transform
from core.types import Matrix4x4

@dataclass(frozen=True)
class FrameSequence:
    frames: Sequence[Mapping[str, Matrix4x4]]
    times: Sequence[float]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("FrameSequence requires at least one frame.")
        if len(self.times) != len(self.frames):
            raise ValueError("FrameSequence.times must have the same length as frames.")
        if any(t < 0.0 for t in self.times):
            raise ValueError("FrameSequence.times must be >= 0.")
        if any(t1 <= t0 for t0, t1 in zip(self.times, self.times[1:])):
            raise ValueError("FrameSequence.times must be strictly increasing.")


@dataclass
class VisualObject:
    name: str
    base_mesh: o3d.geometry.TriangleMesh
    mesh: o3d.geometry.TriangleMesh
    current_transform: np.ndarray
    frame_mesh: o3d.geometry.TriangleMesh | None = None
    frame_size: float | None = None


@dataclass
class TraceObject:
    target_name: str
    line_set: o3d.geometry.LineSet
    color: np.ndarray
    max_points: int
    points: list[np.ndarray]


class Viewer:
    def __init__(
        self,
        window_name: str = "Robot Viewer",
        width: int = 1420,
        height: int = 1080,
        world_frame_size: float = 100.0,
    ):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)

        self._objects: dict[str, VisualObject] = {}
        self._traces: dict[str, TraceObject] = {}
        self._world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=world_frame_size)
        self.vis.add_geometry(self._world_frame)
        self._closed = False

    def add_mesh(
        self,
        name: str,
        mesh: o3d.geometry.TriangleMesh,
        transform: Matrix4x4,
        show_frame: bool = False,
        frame_size: float = 50.0,
    ) -> None:
        if name in self._objects:
            raise ValueError(f"Mesh '{name}' already exists.")

        validate_homogeneous_transform(transform)

        T = np.asarray(transform, dtype=float)

        base_mesh = copy.deepcopy(mesh)
        base_mesh.compute_vertex_normals()

        current_mesh = copy.deepcopy(base_mesh)
        current_mesh.transform(T)

        frame_mesh = None
        if show_frame:
            frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
            frame_mesh.transform(T)

        self._objects[name] = VisualObject(
            name=name,
            base_mesh=base_mesh,
            mesh=current_mesh,
            current_transform=T.copy(),
            frame_mesh=frame_mesh,
            frame_size=frame_size if show_frame else None,
        )

        self.vis.add_geometry(current_mesh)
        if frame_mesh is not None:
            self.vis.add_geometry(frame_mesh)

    def add_robot(
        self,
        robot,
        show_frames: bool = False,
        frame_size: float = 50.0,
    ) -> None:
        for link in robot.links:
            self.add_mesh(
                name=link.name,
                mesh=robot.load_mesh(link.name),
                transform=link.home_frame,
                show_frame=show_frames,
                frame_size=frame_size,
            )

    def add_trace(
        self,
        target_name: str,
        color: tuple[float, float, float] = (0.0, 1.0, 0.0),
        max_points: int = 5000,
    ) -> None:
        if target_name in self._traces:
            raise ValueError(f"Trace for '{target_name}' already exists.")
        if max_points < 2:
            raise ValueError("max_points must be at least 2.")

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        self._traces[target_name] = TraceObject(
            target_name=target_name,
            line_set=line_set,
            color=np.asarray(color, dtype=float),
            max_points=max_points,
            points=[],
        )
        self.vis.add_geometry(line_set)

    def _set_transform(self, name: str, transform: Matrix4x4) -> None:
        if name not in self._objects:
            raise KeyError(f"Unknown mesh '{name}'.")

        validate_homogeneous_transform(transform)

        visual_object = self._objects[name]
        T_old = visual_object.current_transform
        T_new = np.asarray(transform, dtype=float)

        T_delta = T_new @ np.linalg.inv(T_old)

        visual_object.mesh.transform(T_delta)
        if visual_object.frame_mesh is not None:
            visual_object.frame_mesh.transform(T_delta)

        visual_object.current_transform = T_new.copy()

    def set_transforms(self, transforms: Mapping[str, Matrix4x4]) -> None:
        for name, transform in transforms.items():
            if name not in self._objects:
                continue
            self._set_transform(name, transform)

        self._update_traces(transforms)

    def _update_traces(self, transforms: Mapping[str, Matrix4x4]) -> None:
        for target_name, trace in self._traces.items():
            transform = transforms.get(target_name)
            if transform is None:
                continue

            trace.points.append(np.asarray(transform[:3, 3], dtype=float).reshape(3,))
            if len(trace.points) > trace.max_points:
                trace.points = trace.points[-trace.max_points:]

            if len(trace.points) < 2:
                continue

            points = np.vstack(trace.points)
            lines = np.column_stack(
                [np.arange(len(points) - 1), np.arange(1, len(points))]
            ).astype(np.int32)
            colors = np.tile(trace.color, (len(lines), 1))

            trace.line_set.points = o3d.utility.Vector3dVector(points)
            trace.line_set.lines = o3d.utility.Vector2iVector(lines)
            trace.line_set.colors = o3d.utility.Vector3dVector(colors)

    def tick(self) -> bool:
        for visual_object in self._objects.values():
            self.vis.update_geometry(visual_object.mesh)
            if visual_object.frame_mesh is not None:
                self.vis.update_geometry(visual_object.frame_mesh)

        for trace in self._traces.values():
            self.vis.update_geometry(trace.line_set)

        is_open = self.vis.poll_events()
        self.vis.update_renderer()
        return is_open

    def run_callback(self, update_callback: Callable[[float], None], fps: float = 60.0) -> None:
        if fps <= 0.0:
            raise ValueError("fps must be > 0.")

        dt = 1.0 / fps
        t0 = time.perf_counter()
        t_next = t0

        while True:
            now = time.perf_counter()
            if now < t_next:
                time.sleep(t_next - now)
                continue

            update_callback(now - t0)
            if not self.tick():
                break

            t_next += dt

        self.close()

    def run_sequence(self, sequence: FrameSequence, loop: bool = False) -> None:
        frame_delays = [0.0]
        for i in range(1, len(sequence.times)):
            frame_delays.append(float(sequence.times[i] - sequence.times[i - 1]))

        while True:
            for frame, delay in zip(sequence.frames, frame_delays):
                frame_start = time.perf_counter()
                self.set_transforms(frame)
                if not self.tick():
                    self.close()
                    return
                if delay > 0.0:
                    elapsed = time.perf_counter() - frame_start
                    remaining = delay - elapsed
                    if remaining > 0.0:
                        time.sleep(remaining)
            if not loop:
                break
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.vis.destroy_window()
            self._closed = True
