import numpy as np
import open3d as o3d
from core.types import Vector3
from .base import Geometry
from .math3d import segment_point_distance  # your existing helper

class Sphere(Geometry):
    def __init__(self, center_local: Vector3, radius: float, T_world=None, resolution: int = 16):
        super().__init__(T_world=T_world)
        self.c_local = np.asarray(center_local, float)
        self.r = float(radius)
        self._res = int(resolution)

    def _build_mesh_local(self) -> o3d.geometry.TriangleMesh:
        m = o3d.geometry.TriangleMesh.create_sphere(radius=self.r, resolution=self._res)
        m.translate(self.c_local)  # center in LOCAL frame
        return m

    def collides_capsule(self, a_world: Vector3, b_world: Vector3, r: float) -> bool:
        # sphere center in WORLD:
        c_world = self.transform_point_local_to_world(self.c_local)
        d = segment_point_distance(a_world, b_world, c_world)
        return d <= (self.r + float(r))
