
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import open3d as o3d
import numpy as np

from core.types import Vector6
from utils.transformation_helper import translate, rotate, assemble_T
from geometry import Geometry, Capsule, Sphere, Box

@dataclass(frozen=True)
class JointLimit:
    min_pos: np.float64
    max_pos: np.float64
    min_vel: np.float64
    max_vel: np.float64
    min_acc: np.float64
    max_acc: np.float64

@dataclass
class Joint:
    name: str
    mesh_path: str
    mesh: o3d.geometry.TriangleMesh = field(init=False)
    collision_body: Geometry = None
    collision_mesh: o3d.geometry.TriangleMesh = None
    limit: Optional[JointLimit] = None
    T_world: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    
    def __post_init__(self) -> None:
        self.T_world = np.asarray(self.T_world, dtype=float)
        if self.T_world.shape != (4, 4):
            raise ValueError(f"{self.name}: T_world must be 4x4")

        m = o3d.io.read_triangle_mesh(self.mesh_path)
        if m.is_empty():
            raise FileNotFoundError(f"{self.name}: failed to load mesh: {self.mesh_path}")
        m.compute_vertex_normals()

        self.mesh = m
        if self.collision_body:
            self.collision_mesh = self.collision_body.mesh()
            self.collision_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(self.collision_mesh)
            self.collision_mesh.paint_uniform_color([0, 1, 0])  # green wireframe

        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300.0)


    def set_world_transform(self, T_new_world: np.ndarray) -> None:
        """
        Update mesh pose to absolute world transform (applies delta).
        """
        T_new_world = np.asarray(T_new_world, dtype=float)
        if T_new_world.shape != (4, 4):
            raise ValueError(f"{self.name}: T_new_world must be 4x4")

        T_delta = T_new_world @ np.linalg.inv(self.T_world)
        self.mesh.transform(T_delta)

        if self.collision_body:
            self.collision_body.transform(T_delta)
            self.collision_mesh.transform(T_delta)

        self.frame.transform(T_delta)
        self.T_world = T_new_world

    def set_color(self, rgb: List[float]) -> None:
        self.mesh.paint_uniform_color(rgb)