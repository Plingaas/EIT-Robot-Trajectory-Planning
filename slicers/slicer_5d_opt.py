from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pyvista as pv
import trimesh

from .slicer import BaseSlicer


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return (n,3) roughly-uniform unit vectors."""
    # deterministic
    i = np.arange(n, dtype=np.float64)
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    v = np.stack([x, y, z], axis=1)
    # already unit-ish, normalize to be safe
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _overhang_area_score(
    mesh: trimesh.Trimesh,
    up: np.ndarray,
    crit_deg: float = 45.0,
) -> float:
    """
    Sum area of faces that likely need support for a given build direction `up` (unit vector).
    Overhang faces defined as n·up < -cos(crit).
    """
    up = np.asarray(up, dtype=np.float64)
    up /= np.linalg.norm(up)

    n = mesh.face_normals  # (F,3), unit
    a = mesh.area_faces  # (F,)

    thr = -np.cos(np.deg2rad(crit_deg))
    mask = (n @ up) < thr
    return float(a[mask].sum())


def _height_along(mesh: trimesh.Trimesh, up: np.ndarray) -> float:
    """Height of mesh projected along `up`."""
    up = np.asarray(up, dtype=np.float64)
    up /= np.linalg.norm(up)
    proj = mesh.vertices @ up
    return float(proj.max() - proj.min())


def find_best_build_direction(
    mesh: trimesh.Trimesh,
    samples: int = 800,
    crit_deg: float = 45.0,
    height_weight: float = 0.00,
) -> np.ndarray:
    """
    Brute-ish search over sphere directions.
    Minimizes: overhang_area + height_weight * height
    """
    dirs = _fibonacci_sphere(samples)

    best_u = dirs[0]
    best_cost = np.inf

    for u in dirs:
        overhang = _overhang_area_score(mesh, u, crit_deg=crit_deg)
        if height_weight != 0.0:
            overhang += height_weight * _height_along(mesh, u)
        if overhang < best_cost:
            best_cost = overhang
            best_u = u

    return best_u


def _polyline_to_segments(P: np.ndarray) -> np.ndarray:
    if P.shape[0] < 2:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.stack([P[:-1], P[1:]], axis=1)


def _segments_to_pyvista_polydata_with_5d(segments: np.ndarray) -> pv.PolyData:
    if segments.size == 0:
        return pv.PolyData()

    pts = segments.reshape(-1, 3)

    n_seg = segments.shape[0]
    idx = np.arange(0, 2 * n_seg, dtype=np.int64).reshape(-1, 2)
    lines = np.hstack([np.full((n_seg, 1), 2, dtype=np.int64), idx]).ravel()

    poly = pv.PolyData(pts)
    poly.lines = lines

    d = segments[:, 1, :] - segments[:, 0, :]
    heading = np.arctan2(d[:, 1], d[:, 0]).astype(np.float32)  # xy-heading
    tilt = np.zeros((n_seg,), dtype=np.float32)  # placeholder

    heading_pts = np.repeat(heading, 2)
    tilt_pts = np.repeat(tilt, 2)

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


class Slicer5DOptimized(BaseSlicer):
    """
    "5D" slicer with *orientation optimization* to reduce overhangs.
    - Chooses an optimal build direction (up vector) minimizing unsupported overhang area.
    - Slices perpendicular to that direction.
    - Returns contour lines as PolyData, with 5D metadata.
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        dz: float = 0.2,
        crit_deg: float = 45.0,
        samples: int = 800,
        height_weight: float = 0.00,
    ):
        self.mesh = mesh
        self.dz = float(dz)

        # Find "best" build direction
        self.up = find_best_build_direction(
            mesh,
            samples=samples,
            crit_deg=crit_deg,
            height_weight=height_weight,
        )

        # Precompute slice positions along up direction
        proj = self.mesh.vertices @ self.up
        smin, smax = float(proj.min()), float(proj.max())
        ss = np.arange(smin, smax + 1e-9, self.dz)
        self._ss = list(map(float, ss))  # distances along up
        self._i = 0

    @property
    def num_slices(self) -> int:
        return len(self._ss)

    @property
    def zs(self) -> List[float]:
        # Keep the interface name "zs" even though it's "s along up"
        return self._ss

    def slice(self) -> pv.PolyData | None:
        if self._i >= len(self._ss):
            return None

        s = self._ss[self._i]
        self._i += 1

        # Plane: normal = up, origin = up*s (any point with dot(up, origin)=s works)
        origin = self.up * float(s)

        section = self.mesh.section(
            plane_origin=origin.tolist(),
            plane_normal=self.up.tolist(),
        )
        if section is None or not section.discrete:
            return pv.PolyData()

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
