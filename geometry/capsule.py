from __future__ import annotations
import numpy as np
import open3d as o3d

from core.types import Vector3
from .base import Geometry
from .math3d import segment_segment_distance, R_from_z_to_vec


def make_capsule_mesh_z(radius: float, length: float, resolution: int = 24) -> o3d.geometry.TriangleMesh:
    """
    Capsule aligned with +Z, with end-sphere centers at z=0 and z=length.
    Cylinder spans [0, length]. Total end-to-end length = length + 2*radius.
    """
    r = float(radius)
    L = float(max(0.0, length))
    res = int(max(6, resolution))

    # Cylinder in Open3D is centered at origin spanning [-L/2, +L/2] in z, so shift to [0, L]
    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=r, height=max(L, 1e-9), resolution=res, split=4
    )
    cyl.translate((0.0, 0.0, L / 2.0))

    # Build a sphere and cut it into hemispheres by triangle centroid z
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=res)
    V = np.asarray(sph.vertices)
    Tri = np.asarray(sph.triangles)

    tri_centroids = V[Tri].mean(axis=1)  # (Nt,3)
    top_mask = tri_centroids[:, 2] >= 0.0
    bot_mask = tri_centroids[:, 2] <= 0.0

    top = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V.copy()),
        triangles=o3d.utility.Vector3iVector(Tri[top_mask].copy()),
    )
    top.remove_unreferenced_vertices()
    top.translate((0.0, 0.0, L))   # move to end center at z=L

    bot = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V.copy()),
        triangles=o3d.utility.Vector3iVector(Tri[bot_mask].copy()),
    )
    bot.remove_unreferenced_vertices()
    bot.translate((0.0, 0.0, 0.0))  # end center at z=0

    mesh = cyl + top + bot
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


class Capsule(Geometry):
    """
    Capsule defined in LOCAL coordinates by endpoints a_local->b_local (centers of the end hemispheres)
    and radius r. World placement is handled ONLY by T_world (inherited from Geometry).

    Important:
      - We DO NOT override transform().
      - a_local/b_local never change.
      - World endpoints are computed on demand: a_w = T_world * a_local, b_w = T_world * b_local.
    """

    def __init__(
        self,
        a_local: Vector3,
        b_local: Vector3,
        radius: float,
        resolution: int = 24,
        T_world: np.ndarray | None = None,
    ):
        super().__init__(T_world=T_world)
        self.a_local = np.asarray(a_local, float).reshape(3,)
        self.b_local = np.asarray(b_local, float).reshape(3,)
        self.r = float(radius)
        self._res = int(resolution)

    def endpoints_world(self) -> tuple[np.ndarray, np.ndarray]:
        a_w = (self.T_world[:3, :3] @ self.a_local) + self.T_world[:3, 3]
        b_w = (self.T_world[:3, :3] @ self.b_local) + self.T_world[:3, 3]
        return a_w, b_w

    def collides_capsule(self, a_world: Vector3, b_world: Vector3, r: float) -> bool:
        a0, b0 = self.endpoints_world()
        d = segment_segment_distance(a0, b0, a_world, b_world)
        return d <= (self.r + float(r))

    def _build_mesh_local(self) -> o3d.geometry.TriangleMesh:
        """
        Build capsule mesh in LOCAL coords, matching a_local->b_local.
        We build a +Z capsule then rigid-transform it onto the segment a_local->b_local.
        """
        v = self.b_local - self.a_local
        L = float(np.linalg.norm(v))

        mesh = make_capsule_mesh_z(radius=self.r, length=L, resolution=self._res)

        T = np.eye(4)
        T[:3, 3] = self.a_local
        if L > 1e-12:
            T[:3, :3] = R_from_z_to_vec(v)
        mesh.transform(T)

        return mesh

    @staticmethod
    def capsule_from_mesh_pca(mesh: o3d.geometry.TriangleMesh):
        """
        Returns (a_local, b_local, r) in the mesh's CURRENT coordinate frame.
        a_local,b_local are 3-vectors (endpoints), r is scalar.
        """
        V = np.asarray(mesh.vertices, dtype=float)
        if V.shape[0] < 4:
            raise ValueError("Mesh has too few vertices for a stable capsule fit")

        mu = V.mean(axis=0)
        X = V - mu

        C = (X.T @ X) / max(1, X.shape[0])
        eigvals, eigvecs = np.linalg.eigh(C)
        u = eigvecs[:, np.argmax(eigvals)]
        u = u / (np.linalg.norm(u) + 1e-12)

        s = X @ u
        smin, smax = float(s.min()), float(s.max())
        a = mu + u * smin
        b = mu + u * smax

        closest = mu + np.outer(s, u)
        radial = np.linalg.norm(V - closest, axis=1)
        r = float(radial.max())

        return a, b, r
