from __future__ import annotations
import numpy as np
import open3d as o3d

from core.types import Vector3
from .base import Geometry
from .math3d import point_aabb_distance

class Box(Geometry):
    """
    Axis-aligned box in WORLD defined by lo/hi.
    Mesh local frame: corner at self.pos, size is l * w * h
    """
    def __init__(self, pos: Vector3, size: Vector3):
        super().__init__()
        self.pos = np.asarray(pos, float)
        self.size = np.asarray(size, float)

        self.lo = self.pos
        self.hi = self.pos + self.size

    def collides_capsule(self, a: Vector3, b: Vector3, r: float) -> bool:
        # Simple conservative check: sample along segment
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        rr = float(r)
        for i in range(26):
            t = i / 25.0
            p = a + t * (b - a)
            if point_aabb_distance(p, self.lo, self.hi) <= rr:
                return True
        return False

    def _build_mesh_local(self) -> o3d.geometry.TriangleMesh:
        # unit cube [0,1]^3 -> shift to be centered at origin

        m = o3d.geometry.TriangleMesh.create_box(*self.size)
        m.translate(self.pos)
        return m

    def T_world_from_local(self) -> np.ndarray:
        center = 0.5 * (self.lo + self.hi)
        size = (self.hi - self.lo)

        T = np.eye(4)
        # scale
        T[0, 0] = size[0]
        T[1, 1] = size[1]
        T[2, 2] = size[2]
        # translate
        T[:3, 3] = center
        return T
