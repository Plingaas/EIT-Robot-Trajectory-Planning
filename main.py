from __future__ import annotations

import sys
from pathlib import Path

import trimesh

from slicers.slicer import BaseSlicer
from slicers.slicer_3d import Slicer3D
from slicers.slicer_5d import Slicer5D
from slicers.slicer_5d_opt import Slicer5DOptimized
from slicers.slicer_nonplanar import NonPlanarHeightfieldSlicer
from slicers.slicer_utils import _trimesh_to_pyvista, slice_all
from vis.viewer import show_slices


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/model.stl")
        return 2

    mesh_path = Path(sys.argv[1]).expanduser().resolve()
    if not mesh_path.exists():
        print(f"File not found: {mesh_path}")
        return 2

    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        print("Loaded geometry is not a Trimesh. Try exporting as STL/OBJ/PLY.")
        return 2

    slicer = NonPlanarHeightfieldSlicer(
        mesh,
        pitch_xy=1.0,  # try 1.0–2.0 first for speed+RAM
        layer_height=0.2,
        max_overhang_deg=45.0,
        relax_iters=20,
    )

    print(f"Slicing... total layers: {slicer.num_slices}")
    slices, zs = slice_all(slicer)
    print(f"Done. Stored {len(slices)} slices (some may be empty).")

    base = _trimesh_to_pyvista(mesh)
    show_slices(base, slices, zs, title="Slicer Debug Viewer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
