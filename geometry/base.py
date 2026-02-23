from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import open3d as o3d
from core.types import Vector3


def _apply_T_to_vertices(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    V = np.asarray(V, float)
    T = np.asarray(T, float)
    ones = np.ones((V.shape[0], 1))
    Vh = np.hstack([V, ones])          # Nx4
    Vw = (T @ Vh.T).T[:, :3]           # Nx3
    return Vw

class Geometry(ABC):
    """
    Base class for all geometry / obstacles.
    - Owns an Open3D mesh (stable object).
    - Owns a world transform T_world.
    - transform(T_delta) behaves like Open3D: apply delta on top of current pose.
    """

    def __init__(self, T_world: np.ndarray | None = None):
        self.T_world = np.eye(4) if T_world is None else np.asarray(T_world, float)
        if self.T_world.shape != (4, 4):
            raise ValueError("T_world must be 4x4")

        self._mesh: o3d.geometry.TriangleMesh | None = None
        self._V_local: np.ndarray | None = None  # base vertices in local frame

    @abstractmethod
    def collides_capsule(self, a_world: Vector3, b_world: Vector3, r: float) -> bool:
        ...

    @abstractmethod
    def _build_mesh_local(self) -> o3d.geometry.TriangleMesh:
        """Create mesh in LOCAL coordinates (no world transform applied)."""
        ...

    def mesh(self) -> o3d.geometry.TriangleMesh:
        if self._mesh is None:
            self._mesh = self._build_mesh_local()
            self._mesh.compute_vertex_normals()
            self._V_local = np.asarray(self._mesh.vertices).copy()
            # apply initial pose
            self._update_mesh_vertices()
            return self._mesh
        return self._mesh

    def _update_mesh_vertices(self) -> None:
        if self._mesh is None:
            return
        Vw = _apply_T_to_vertices(self._V_local, self.T_world)
        self._mesh.vertices = o3d.utility.Vector3dVector(Vw)
        self._mesh.compute_vertex_normals()

    def transform(self, T_delta: np.ndarray) -> None:
        """
        Like Open3D: apply delta transform on top of current pose.
        """
        T_delta = np.asarray(T_delta, float)
        if T_delta.shape != (4, 4):
            raise ValueError("T_delta must be 4x4")
        self.T_world = T_delta @ self.T_world
        self._update_mesh_vertices()

    def set_world_transform(self, T_new_world: np.ndarray) -> None:
        """
        Set absolute pose, Open3D-style by applying delta internally.
        """
        T_new_world = np.asarray(T_new_world, float)
        if T_new_world.shape != (4, 4):
            raise ValueError("T_new_world must be 4x4")
        T_delta = T_new_world @ np.linalg.inv(self.T_world)
        self.transform(T_delta)

    def transform_point_local_to_world(self, p_local: Vector3) -> Vector3:
        p_local = np.asarray(p_local, float).reshape(3,)
        return (self.T_world[:3, :3] @ p_local) + self.T_world[:3, 3]
