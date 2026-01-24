from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pyvista as pv


@dataclass
class ViewState:
    current_idx: int = 0
    show_mesh: bool = True
    show_up_to_current: bool = False  # False => only current slice visible


def _apply_visibility(
    state: ViewState,
    slice_actors,
    mesh_actor,
    text_actor,
    zs: List[float],
    valid_indices: List[int],
):

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
        real_i = valid_indices[i]  # map actor index -> slice index
        text_actor.SetText(0, f"Layer {real_i}/{len(zs)-1}   s={zs[real_i]:.4f}")


def show_slices(
    base_mesh_polydata: pv.PolyData,
    slices: List[pv.PolyData],
    zs: List[float],
    title: str = "Slicer Debug Viewer",
) -> None:
    plotter = pv.Plotter(title=title)
    state = ViewState(current_idx=0, show_mesh=True, show_up_to_current=False)

    # Base mesh (transparent)
    mesh_actor = plotter.add_mesh(base_mesh_polydata, opacity=0.20)

    # Slice actors
    slice_actors = []
    valid_indices = []  # map actor index -> slice index

    for i, poly in enumerate(slices):
        if poly.n_points == 0:
            continue
        actor = plotter.add_mesh(poly, line_width=2, opacity=1.0)
        actor.SetVisibility(1 if len(slice_actors) == 0 else 0)
        slice_actors.append(actor)
        valid_indices.append(i)

    plotter.add_axes()
    plotter.show_grid()

    text_actor = plotter.add_text("", position="upper_left")
    _apply_visibility(state, slice_actors, mesh_actor, text_actor, zs, valid_indices)

    # --- Slider (continuous update via VTK InteractionEvent)
    last_idx = {"i": -1}

    def on_slider_value(val: float):
        idx = int(round(val))
        idx = max(0, min(idx, len(slice_actors) - 1))
        if idx == last_idx["i"]:
            return
        last_idx["i"] = idx

        state.current_idx = idx
        _apply_visibility(
            state, slice_actors, mesh_actor, text_actor, zs, valid_indices
        )
        plotter.render()

    slider = plotter.add_slider_widget(
        callback=lambda v: None,
        rng=(0, max(0, len(slice_actors) - 1)),
        value=0,
        title="Layer index",
        fmt="%.0f",
    )

    def _on_interaction(widget, event):
        rep = widget.GetRepresentation()
        on_slider_value(rep.GetValue())

    slider.AddObserver("InteractionEvent", _on_interaction)
    slider.AddObserver("EndInteractionEvent", _on_interaction)
    on_slider_value(0)

    # --- Checkboxes
    def on_mesh_toggle(checked: bool):
        state.show_mesh = bool(checked)
        _apply_visibility(
            state, slice_actors, mesh_actor, text_actor, zs, valid_indices
        )
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

    def on_up_to_toggle(checked: bool):
        state.show_up_to_current = bool(checked)
        _apply_visibility(
            state, slice_actors, mesh_actor, text_actor, zs, valid_indices
        )
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

    # --- Hotkeys
    def toggle_mesh_key():
        state.show_mesh = not state.show_mesh
        _apply_visibility(
            state, slice_actors, mesh_actor, text_actor, zs, valid_indices
        )
        plotter.render()

    def toggle_accum_key():
        state.show_up_to_current = not state.show_up_to_current
        _apply_visibility(
            state, slice_actors, mesh_actor, text_actor, zs, valid_indices
        )
        plotter.render()

    def prev_layer():
        state.current_idx = max(0, state.current_idx - 1)
        _apply_visibility(
            state, slice_actors, mesh_actor, text_actor, zs, valid_indices
        )
        plotter.render()

    def next_layer():
        state.current_idx = min(len(slice_actors) - 1, state.current_idx + 1)
        _apply_visibility(
            state, slice_actors, mesh_actor, text_actor, zs, valid_indices
        )
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
