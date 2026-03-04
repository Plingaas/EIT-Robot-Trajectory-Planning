import copy
import time
import numpy as np
import open3d as o3d

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from core.types import Matrix4x4


@dataclass(frozen=True)
class FrameSequence:
    frames: Sequence[Mapping[str, Matrix4x4]]
    fps: float = 60.0
    loop: bool = False

    def __post_init__(self) -> None:
        if self.fps <= 0.0:
            raise ValueError("FrameSequence.fps must be > 0.")
        if not self.frames:
            raise ValueError("FrameSequence requires at least one frame.")


@dataclass
class _VisualObject:
    name: str
    base_mesh: o3d.geometry.TriangleMesh
    mesh: o3d.geometry.TriangleMesh
    frame_mesh: o3d.geometry.TriangleMesh | None = None
    frame_size: float | None = None


def _validate_transform(transform: Matrix4x4, name: str) -> Matrix4x4:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name}: expected a 4x4 transform, got {transform.shape}.")
    return transform


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

        self._objects: dict[str, _VisualObject] = {}
        self._world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=world_frame_size)
        self.vis.add_geometry(self._world_frame)
        self._closed = False

    def add_mesh(
        self,
        name: str,
        mesh: o3d.geometry.TriangleMesh,
        transform: Matrix4x4 | None = None,
        show_frame: bool = False,
        frame_size: float = 50.0,
    ) -> None:
        if name in self._objects:
            raise ValueError(f"Mesh '{name}' already exists.")

        initial_transform = np.eye(4, dtype=float) if transform is None else _validate_transform(transform, name)
        base_mesh = copy.deepcopy(mesh)
        base_mesh.compute_vertex_normals()

        current_mesh = copy.deepcopy(base_mesh)
        current_mesh.transform(initial_transform)

        frame_mesh = None
        if show_frame:
            frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
            frame_mesh.transform(initial_transform)

        self._objects[name] = _VisualObject(
            name=name,
            base_mesh=base_mesh,
            mesh=current_mesh,
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

    def _set_transform(self, name: str, transform: Matrix4x4) -> None:
        if name not in self._objects:
            raise KeyError(f"Unknown mesh '{name}'.")
        visual_object = self._objects[name]
        transform = _validate_transform(transform, name=name)

        self.vis.remove_geometry(visual_object.mesh, reset_bounding_box=False)
        mesh = copy.deepcopy(visual_object.base_mesh)
        mesh.transform(transform)
        visual_object.mesh = mesh
        self.vis.add_geometry(visual_object.mesh, reset_bounding_box=False)

        if visual_object.frame_mesh is not None:
            self.vis.remove_geometry(visual_object.frame_mesh, reset_bounding_box=False)
            frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=50.0 if visual_object.frame_size is None else visual_object.frame_size
            )
            frame_mesh.transform(transform)
            visual_object.frame_mesh = frame_mesh
            self.vis.add_geometry(visual_object.frame_mesh, reset_bounding_box=False)

    def set_transforms(self, transforms: Mapping[str, Matrix4x4]) -> None:
        missing = [name for name in transforms if name not in self._objects]
        if missing:
            raise KeyError(f"Unknown meshes in frame: {missing}")

        for name, transform in transforms.items():
            self._set_transform(name, transform)

    def tick(self) -> bool:
        for visual_object in self._objects.values():
            self.vis.update_geometry(visual_object.mesh)
            if visual_object.frame_mesh is not None:
                self.vis.update_geometry(visual_object.frame_mesh)
        is_open = self.vis.poll_events()
        self.vis.update_renderer()
        return is_open

    def is_open(self) -> bool:
        if self._closed:
            return False
        return self.vis.poll_events()

    def run(
        self,
        sequence: FrameSequence | None = None,
        update_callback: Callable[[float], None] | None = None,
        fps: float = 60.0,
    ) -> None:
        if sequence is not None and update_callback is not None:
            raise ValueError("Use either 'sequence' or 'update_callback', not both.")
        if sequence is None and update_callback is None:
            raise ValueError("Viewer.run() requires either 'sequence' or 'update_callback'.")

        if sequence is not None:
            self._run_sequence(sequence)
            return

        self._run_callback(update_callback=update_callback, fps=fps)

    def _run_sequence(self, sequence: FrameSequence) -> None:
        dt = 1.0 / sequence.fps

        while True:
            for frame in sequence.frames:
                frame_start = time.perf_counter()
                self.set_transforms(frame)
                if not self.tick():
                    self.close()
                    return
                elapsed = time.perf_counter() - frame_start
                remaining = dt - elapsed
                if remaining > 0.0:
                    time.sleep(remaining)

            if not sequence.loop:
                break

        while self.tick():
            time.sleep(0.01)

        self.close()

    def _run_callback(self, update_callback: Callable[[float], None], fps: float = 60.0) -> None:
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

    def close(self) -> None:
        if not self._closed:
            self.vis.destroy_window()
            self._closed = True
