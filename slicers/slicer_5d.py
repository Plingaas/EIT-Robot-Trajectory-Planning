from __future__ import annotations

from typing import List

import numpy as np
import pyvista as pv
import trimesh

from .slicer import BaseSlicer


def _polyline_to_segments(P: np.ndarray) -> np.ndarray:
    """
    P: (N,3) polyline points
    returns segments: (N-1,2,3)
    """
    if P.shape[0] < 2:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.stack([P[:-1], P[1:]], axis=1)


def _segments_to_pyvista_polydata_with_5d(segments: np.ndarray) -> pv.PolyData:
    """
    segments: (M,2,3)

    Returns a PolyData with:
      - points: (2M,3)
      - lines: connectivity
      - point_data:
          heading_rad: (2M,)
          tilt_rad:    (2M,)
          tool_axis:   (2M,3)
    """
    if segments.size == 0:
        return pv.PolyData()

    # Points for PyVista
    pts = segments.reshape(-1, 3)  # (2M,3)

    # VTK line cells: [2, i0, i1, 2, i2, i3, ...]
    n_seg = segments.shape[0]
    idx = np.arange(0, 2 * n_seg, dtype=np.int64).reshape(-1, 2)
    lines = np.hstack([np.full((n_seg, 1), 2, dtype=np.int64), idx]).ravel()

    poly = pv.PolyData(pts)
    poly.lines = lines

    # --- 5D attributes (heading, tilt)
    # For each segment, compute heading in XY: atan2(dy, dx)
    d = segments[:, 1, :] - segments[:, 0, :]  # (M,3)
    heading = np.arctan2(d[:, 1], d[:, 0]).astype(np.float32)  # (M,)
    tilt = np.zeros((n_seg,), dtype=np.float32)  # placeholder: tool points up

    # Each segment has 2 endpoints => duplicate scalars for endpoints
    heading_pts = np.repeat(heading, 2)  # (2M,)
    tilt_pts = np.repeat(tilt, 2)  # (2M,)

    # Tool axis vector from heading+tilt:
    # tilt=0 => axis = +Z always; keep formula so it generalizes later
    # axis = [sin(tilt)*cos(heading), sin(tilt)*sin(heading), cos(tilt)]
    sin_t = np.sin(tilt_pts)
    tool_axis = np.column_stack(
        [
            sin_t * np.cos(heading_pts),
            sin_t * np.sin(heading_pts),
            np.cos(tilt_pts),
        ]
    ).astype(np.float32)

    poly.point_data["heading_rad"] = heading_pts
    poly.point_data["tilt_rad"] = tilt_pts
    poly.point_data["tool_axis"] = tool_axis

    return poly


class Slicer5D(BaseSlicer):
    """
    5D slicer (position + heading + tilt).

    For now:
      - Geometry is the same slice contours as Slicer3D (Z-plane section).
      - Adds per-point "heading_rad" and "tilt_rad" (tilt=0 placeholder)
      - Adds per-point "tool_axis" vector derived from heading+tilt
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

    def slice(self) -> pv.PolyData | None:
        if self._i >= len(self._zs):
            return None

        z = self._zs[self._i]
        self._i += 1

        section = self.mesh.section(
            plane_origin=[0, 0, float(z)],
            plane_normal=[0, 0, 1],
        )

        if section is None or not section.discrete:
            return pv.PolyData()

        # Combine all polylines into one segments array
        segments_all = []
        for P in section.discrete:
            P = np.asarray(P)
            seg = _polyline_to_segments(P)
            if seg.size:
                segments_all.append(seg)

        if not segments_all:
            return pv.PolyData()

        segments = np.concatenate(segments_all, axis=0)
        return _segments_to_pyvista_polydata_with_5d(segments)
