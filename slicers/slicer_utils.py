from __future__ import annotations

from typing import List, Tuple

from .slicer import BaseSlicer

import numpy as np
import pyvista as pv
import trimesh


def slice_all(slicer: BaseSlicer) -> Tuple[List[pv.PolyData], List[float]]:
    zs = slicer.zs
    out: List[pv.PolyData] = []
    while True:
        s = slicer.slice()
        if s is None:
            break
        out.append(s)
    return out, zs


def _trimesh_to_pyvista(mesh: trimesh.Trimesh):

    faces = np.hstack(
        [
            np.full((mesh.faces.shape[0], 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64),
        ]
    ).ravel()
    return pv.PolyData(mesh.vertices, faces)


def _segments_to_pyvista_polydata(segments: np.ndarray):
    """
    segments: (M,2,3) float array of line segments in 3D
    returns: pyvista.PolyData with line cells
    """
    import pyvista as pv

    if segments.size == 0:
        return pv.PolyData()

    pts = segments.reshape(-1, 3)  # (2M,3)

    n_seg = segments.shape[0]
    idx = np.arange(0, 2 * n_seg, dtype=np.int64).reshape(-1, 2)
    lines = np.hstack([np.full((n_seg, 1), 2, dtype=np.int64), idx]).ravel()

    poly = pv.PolyData(pts)
    poly.lines = lines
    return poly
