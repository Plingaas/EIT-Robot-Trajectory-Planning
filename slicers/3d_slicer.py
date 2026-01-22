# Local includes
from .slicer import BaseSlicer

# Library includes
import trimesh
import numpy as np


class Slicer3D(BaseSlicer):
    """
    Simple Z-plane slicer: produces per-layer contour lines.
    """

    def __init__(self, mesh: trimesh.Trimesh, dz: float = 0.2):
        self.mesh = mesh
        self.dz = float(dz)

        zmin, zmax = self.mesh.bounds[:, 2]
        zs = np.arange(zmin, zmax + 1e-9, self.dz)
        self._zs = list(map(float, zs))
        self._i = 0

    @property
    def num_slices(self) -> int:
        return len(self._zs)

    @property
    def zs(self) -> List[float]:
        return self._zs

    def slice(self):
        import pyvista as pv

        if self._i >= len(self._zs):
            return None

        z = self._zs[self._i]
        self._i += 1

        section = self.mesh.section(
            plane_origin=[0, 0, float(z)],
            plane_normal=[0, 0, 1],
        )
        if section is None or not section.discrete:
            return pv.PolyData()  # empty slice

        segments_all = []
        for P in section.discrete:
            if len(P) < 2:
                continue
            segments_all.append(np.stack([P[:-1], P[1:]], axis=1))  # (M,2,3)

        if not segments_all:
            return pv.PolyData()

        segments = np.concatenate(segments_all, axis=0)
        return _segments_to_pyvista_polydata(segments)
