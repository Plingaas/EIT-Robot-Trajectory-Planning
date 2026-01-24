from __future__ import annotations

from typing import List, Optional
import numpy as np
import pyvista as pv
import trimesh

from .slicer import BaseSlicer


def _heightfield_to_surface_polydata_masked(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, solid_mask: np.ndarray
) -> pv.PolyData:
    """
    Keep only cells where all 4 corners are inside the solid_mask.
    """
    X, Y = np.meshgrid(x, y, indexing="ij")  # (nx, ny)
    grid = pv.StructuredGrid(X, Y, z)

    # Cell mask: a quad cell is valid if all its 4 corner points are valid
    # solid_mask is (nx, ny) on points
    cell_ok = (
        solid_mask[:-1, :-1]
        & solid_mask[1:, :-1]
        & solid_mask[:-1, 1:]
        & solid_mask[1:, 1:]
    )  # (nx-1, ny-1)

    # Store as cell data and threshold
    grid.cell_data["solid_cell"] = cell_ok.ravel(order="F").astype(np.uint8)
    clipped = grid.threshold(0.5, scalars="solid_cell", preference="cell")

    return clipped.extract_surface()


def _project_to_lipschitz(
    h: np.ndarray, mask: np.ndarray, max_dz: float, iters: int = 25
) -> np.ndarray:
    """
    Enforce |h(i)-h(j)| <= max_dz for 4-neighbors on masked cells.
    (Simple iterative projection; good enough for interactive debug)
    """
    h = h.copy()
    for _ in range(iters):
        # left -> right
        h[1:, :] = np.minimum(h[1:, :], h[:-1, :] + max_dz)
        h[1:, :] = np.maximum(h[1:, :], h[:-1, :] - max_dz)

        # right -> left
        h[:-1, :] = np.minimum(h[:-1, :], h[1:, :] + max_dz)
        h[:-1, :] = np.maximum(h[:-1, :], h[1:, :] - max_dz)

        # down -> up
        h[:, 1:] = np.minimum(h[:, 1:], h[:, :-1] + max_dz)
        h[:, 1:] = np.maximum(h[:, 1:], h[:, :-1] - max_dz)

        # up -> down
        h[:, :-1] = np.minimum(h[:, :-1], h[:, 1:] + max_dz)
        h[:, :-1] = np.maximum(h[:, :-1], h[:, 1:] - max_dz)

        # keep outside mask stable (we'll clamp elsewhere too)
        h[~mask] = 0.0
    return h


class NonPlanarHeightfieldSlicer(BaseSlicer):
    """
    Non-planar slicer that avoids 3D voxel grids.

    - Samples an (x,y) grid.
    - Uses ray casting along +Z to find bottom/top of solid per column.
    - Builds printable layers as a heightfield H(x,y) with a slope constraint.
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        pitch_xy: float = 1.0,  # mm (increase to reduce RAM/compute)
        layer_height: float = 0.2,  # mm
        max_overhang_deg: float = 45.0,
        relax_iters: int = 25,
        use_embree_if_available: bool = True,
    ):
        self.mesh = mesh
        self.pitch_xy = float(pitch_xy)
        self.layer_height = float(layer_height)
        self.max_overhang_deg = float(max_overhang_deg)
        self.relax_iters = int(relax_iters)

        # Optional speedup (no RAM blowup): pyembree accelerator if installed
        if use_embree_if_available:
            try:
                # trimesh will use embree intersector if available
                _ = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)
            except Exception:
                pass

        bounds = self.mesh.bounds  # (2,3)
        self.xmin, self.ymin, self.zmin = bounds[0]
        self.xmax, self.ymax, self.zmax = bounds[1]

        # grid cell centers
        self.x = (
            np.arange(self.xmin, self.xmax + 1e-9, self.pitch_xy, dtype=np.float32)
            + self.pitch_xy * 0.5
        )
        self.y = (
            np.arange(self.ymin, self.ymax + 1e-9, self.pitch_xy, dtype=np.float32)
            + self.pitch_xy * 0.5
        )
        self.nx, self.ny = len(self.x), len(self.y)

        # bottom/top per (x,y) column in world z (float32)
        self.bottom = np.full((self.nx, self.ny), np.nan, dtype=np.float32)
        self.top = np.full((self.nx, self.ny), np.nan, dtype=np.float32)

        self._compute_column_bounds_via_rays()

        self.solid_mask = np.isfinite(self.top)

        # Printed height map H starts at bottom surface (supported by bed assumption)
        # If model floats above bed, you will see that in bottom values.
        self.H = np.where(self.solid_mask, self.bottom, 0.0).astype(np.float32)

        # slope constraint: max delta-z between neighbors
        max_slope = np.tan(np.deg2rad(self.max_overhang_deg))
        self.max_dz = float(max_slope * self.pitch_xy)  # dz per one cell step

        # estimate number of layers
        if self.solid_mask.any():
            max_height = float(np.nanmax(self.top - self.bottom))
            self._num = int(np.ceil(max_height / self.layer_height))
        else:
            self._num = 0

        self._k = 0
        self._zs_cache: Optional[List[float]] = None

    def _compute_column_bounds_via_rays(self):
        """
        For each (x,y), cast a ray from below bbox upward.
        Use odd-even rule: intersections sorted; inside spans are pairs.
        We approximate bottom as first intersection, top as last intersection.
        """
        # Ray origins: one per grid cell
        # Start a bit below zmin to guarantee entry
        z0 = float(self.zmin - 10.0 * self.layer_height - 1.0)
        origins = np.zeros((self.nx * self.ny, 3), dtype=np.float32)
        dirs = np.zeros((self.nx * self.ny, 3), dtype=np.float32)
        dirs[:, 2] = 1.0

        # Fill origins in row-major
        idx = 0
        for ix in range(self.nx):
            for iy in range(self.ny):
                origins[idx, 0] = self.x[ix]
                origins[idx, 1] = self.y[iy]
                origins[idx, 2] = z0
                idx += 1

        # Intersections: returns locations + which ray index
        loc, ray_id, _ = self.mesh.ray.intersects_location(
            origins, dirs, multiple_hits=True
        )

        if len(loc) == 0:
            return

        # Group z intersections per ray id
        # We'll sort by ray_id then z
        z_hits = loc[:, 2].astype(np.float32)
        order = np.lexsort((z_hits, ray_id))
        ray_id = ray_id[order]
        z_hits = z_hits[order]

        # walk through groups
        start = 0
        N = len(z_hits)
        while start < N:
            rid = int(ray_id[start])
            end = start + 1
            while end < N and int(ray_id[end]) == rid:
                end += 1

            zs = z_hits[start:end]
            # remove near-duplicates from grazing hits
            if zs.size >= 2:
                zs = np.unique(np.round(zs, 5))

            if zs.size >= 2:
                # odd-even: inside segments are pairs; for "solid" typical shapes,
                # bottom is first, top is last.
                b = float(zs[0])
                t = float(zs[-1])

                ix = rid // self.ny
                iy = rid % self.ny
                self.bottom[ix, iy] = b
                self.top[ix, iy] = t

            start = end

    @property
    def num_slices(self) -> int:
        return self._num

    @property
    def zs(self) -> List[float]:
        if self._zs_cache is None:
            self._zs_cache = [float(i) for i in range(self._num)]
        return self._zs_cache

    def slice(self) -> pv.PolyData | None:
        if self._k >= self._num or not self.solid_mask.any():
            return None

        # Proposed next height
        target = self.H + self.layer_height
        # Clamp to top
        target = np.minimum(target, self.top)
        # Only where we can still grow
        active = self.solid_mask & (target > self.H + 1e-6)

        if not active.any():
            self._k = self._num
            return None

        h = self.H.copy()
        h[active] = target[active]

        # Enforce slope constraint (supportless-ish)
        h2 = _project_to_lipschitz(
            h, active, max_dz=self.max_dz, iters=self.relax_iters
        )

        # Clamp again to feasible
        h2 = np.minimum(h2, self.top)
        h2 = np.maximum(h2, self.H)
        h2[~self.solid_mask] = 0.0

        self.H = h2.astype(np.float32)
        self._k += 1

        poly = _heightfield_to_surface_polydata_masked(
            self.x, self.y, self.H.astype(np.float32), self.solid_mask
        )
        return poly if poly.n_points > 0 else pv.PolyData()
