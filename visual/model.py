from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d

from core.types import Matrix4x4


@dataclass(frozen=True)
class VisualLink:
    name: str
    mesh_path: Path
    home_frame: Matrix4x4
    color: tuple[float, float, float] | None = None


class VisualRobot:
    def __init__(self, links: Iterable[VisualLink]):
        self._links = tuple(links)
        if not self._links:
            raise ValueError("VisualRobot requires at least one link.")

        names = [link.name for link in self._links]
        if len(set(names)) != len(names):
            raise ValueError("VisualRobot link names must be unique.")

    @property
    def links(self) -> tuple[VisualLink, ...]:
        return self._links

    def link_names(self) -> list[str]:
        return [link.name for link in self._links]

    def default_frames(self) -> dict[str, Matrix4x4]:
        return {link.name: np.array(link.home_frame, dtype=float, copy=True) for link in self._links}

    def load_mesh(self, link_name: str) -> o3d.geometry.TriangleMesh:
        link = next((item for item in self._links if item.name == link_name), None)
        if link is None:
            raise KeyError(f"Unknown link '{link_name}'.")

        mesh = o3d.io.read_triangle_mesh(str(link.mesh_path))
        if mesh.is_empty():
            raise FileNotFoundError(f"{link.name}: failed to load mesh: {link.mesh_path}")

        mesh.compute_vertex_normals()
        if link.color is not None:
            mesh.paint_uniform_color(link.color)
        return mesh

