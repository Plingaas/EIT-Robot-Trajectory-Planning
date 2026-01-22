from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import trimesh

# --------------------------- VIEWER ---------------------------


def _trimesh_to_pyvista(mesh: trimesh.Trimesh):
    import pyvista as pv

    faces = np.hstack(
        [
            np.full((mesh.faces.shape[0], 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64),
        ]
    ).ravel()
    return pv.PolyData(mesh.vertices, faces)


@dataclass
class ViewState:
    current_idx: int = 0
    show_mesh: bool = True
    show_up_to_current: bool = False  # False => only current slice visible


def _apply_visibility(
    state: ViewState, slice_actors, mesh_actor, text_actor, zs: List[float]
):
    # Toggle mesh
    mesh_actor.SetVisibility(1 if state.show_mesh else 0)

    n = len(slice_actors)
    if n == 0:
        return

    i = max(0, min(state.current_idx, n - 1))
    state.current_idx = i

    if state.show_up_to_current:
        for k in range(n):
            slice_actors[k].SetVisibility(1 if k <= i else 0)
    else:
        for k in range(n):
            slice_actors[k].SetVisibility(1 if k == i else 0)

    if text_actor is not None:
        text_actor.SetText(0, f"Layer {i}/{n-1}   z={zs[i]:.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/model.stl")
        raise SystemExit(2)

    mesh_path = Path(sys.argv[1]).expanduser().resolve()
    if not mesh_path.exists():
        print(f"File not found: {mesh_path}")
        raise SystemExit(2)

    # 1) run main on specific filepath
    print(f"Loading: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        print("Loaded geometry is not a Trimesh. Try exporting as STL/OBJ/PLY.")
        raise SystemExit(2)

    # 2) pick slicer class (keep same slicer for now)
    slicer: BaseSlicer = Slicer3D(mesh, dz=0.2)

    # 3) slicer.slice() returns the next slice
    # 4) slice whole model and store in list
    print(f"Slicing... total layers: {slicer.num_slices}")
    zs: List[float] = slicer.zs
    slices = []
    while True:
        s = slicer.slice()
        if s is None:
            break
        slices.append(s)

    print(f"Done. Stored {len(slices)} slices (some may be empty).")

    # 5) display with slider + buttons
    import pyvista as pv

    plotter = pv.Plotter(title="Slicer Debug Viewer")
    state = ViewState(current_idx=0, show_mesh=True, show_up_to_current=False)

    # Base mesh actor (transparent)
    base = _trimesh_to_pyvista(mesh)
    mesh_actor = plotter.add_mesh(
        base, opacity=0.20
    )  # transparent by default; toggled on/off by button

    # Slice actors (one actor per layer; we toggle visibility)
    slice_actors = []
    for i, poly in enumerate(slices):
        # Use one color for now; you can zebra-color by alternating add_mesh params if you want.
        actor = plotter.add_mesh(poly, line_width=2, opacity=1.0)
        actor.SetVisibility(1 if i == 0 else 0)
        slice_actors.append(actor)

    plotter.add_axes()
    plotter.show_grid()

    text_actor = plotter.add_text("", position="upper_left")
    _apply_visibility(state, slice_actors, mesh_actor, text_actor, zs)

    # --- Slider: 0 .. n-1
    def on_slider(val: float):
        state.current_idx = int(round(val))
        _apply_visibility(state, slice_actors, mesh_actor, text_actor, zs)
        plotter.render()

    last_idx = {"i": -1}

    def on_slider_value(val: float):
        idx = int(round(val))
        idx = max(0, min(idx, len(slice_actors) - 1))
        if idx == last_idx["i"]:
            return
        last_idx["i"] = idx

        state.current_idx = idx
        _apply_visibility(state, slice_actors, mesh_actor, text_actor, zs)
        plotter.render()

    # Create slider normally (no event_type)
    slider = plotter.add_slider_widget(
        callback=lambda v: None,  # we'll drive updates ourselves
        rng=(0, max(0, len(slice_actors) - 1)),
        value=0,
        title="Layer index",
        fmt="%.0f",
    )

    # Force continuous updates while dragging:
    def _on_interaction(widget, event):
        rep = widget.GetRepresentation()
        on_slider_value(rep.GetValue())

    slider.AddObserver("InteractionEvent", _on_interaction)

    # (optional) also update on release, harmless:
    slider.AddObserver("EndInteractionEvent", _on_interaction)

    # Initialize
    on_slider_value(0)

    # --- "Buttons": use checkboxes (simple, built-in, reliable)
    # Checkbox 1: show mesh (transparent model on/off)
    def on_mesh_toggle(checked: bool):
        state.show_mesh = bool(checked)
        _apply_visibility(state, slice_actors, mesh_actor, text_actor, zs)
        plotter.render()

    plotter.add_checkbox_button_widget(
        callback=on_mesh_toggle,
        value=True,
        position=(10, 10),
        size=30,
        border_size=2,
        color_on="white",
        color_off="gray",
        background_color="black",
    )
    plotter.add_text("Show mesh", position=(50, 14), font_size=10)

    # Checkbox 2: show all slices up to current (vs only current)
    def on_up_to_toggle(checked: bool):
        state.show_up_to_current = bool(checked)
        _apply_visibility(state, slice_actors, mesh_actor, text_actor, zs)
        plotter.render()

    plotter.add_checkbox_button_widget(
        callback=on_up_to_toggle,
        value=False,
        position=(10, 50),
        size=30,
        border_size=2,
        color_on="white",
        color_off="gray",
        background_color="black",
    )
    plotter.add_text("Show 0..current", position=(50, 54), font_size=10)

    # Optional: keyboard shortcuts (still “buttons”, but handy)
    #   M toggles mesh, A toggles show up-to, left/right changes layer
    def toggle_mesh_key():
        state.show_mesh = not state.show_mesh
        _apply_visibility(state, slice_actors, mesh_actor, text_actor, zs)
        plotter.render()

    def toggle_accum_key():
        state.show_up_to_current = not state.show_up_to_current
        _apply_visibility(state, slice_actors, mesh_actor, text_actor, zs)
        plotter.render()

    def prev_layer():
        state.current_idx = max(0, state.current_idx - 1)
        _apply_visibility(state, slice_actors, mesh_actor, text_actor, zs)
        plotter.render()

    def next_layer():
        state.current_idx = min(len(slice_actors) - 1, state.current_idx + 1)
        _apply_visibility(state, slice_actors, mesh_actor, text_actor, zs)
        plotter.render()

    plotter.add_key_event("m", toggle_mesh_key)
    plotter.add_key_event("a", toggle_accum_key)
    plotter.add_key_event("Left", prev_layer)
    plotter.add_key_event("Right", next_layer)

    print("Controls:")
    print("  - Slider: select layer")
    print("  - Checkbox: Show mesh (transparent) on/off")
    print("  - Checkbox: Show 0..current (accumulate slices)")
    print("  - Keys: M=toggle mesh, A=toggle accumulate, Left/Right=step layers")
    print("Close window to exit.")
    plotter.show()


if __name__ == "__main__":
    main()
