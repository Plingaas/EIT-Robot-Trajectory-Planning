from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from core.kinematics.fk import fk_links
from robot.ur5e_parameters import M_LIST, S
from visual.ur5e_model import LINK_ORDER, UR5e


JOINT_LIMIT = 2.0 * np.pi
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
PANEL_WIDTH = 340

LINK_COLORS = {
    "mount": (0.45, 0.45, 0.45),
    "base": (0.72, 0.72, 0.72),
    "shoulder": (0.86, 0.86, 0.82),
    "elbow": (0.78, 0.82, 0.86),
    "forearm": (0.88, 0.80, 0.70),
    "wrist": (0.70, 0.78, 0.84),
    "end_effector": (0.55, 0.55, 0.58),
}


@dataclass
class JointControl:
    slider: gui.Slider
    value_label: gui.Label


def _parse_q(values: list[float] | None, degrees: bool) -> np.ndarray:
    if values is None:
        return np.zeros(6, dtype=float)
    q = np.asarray(values, dtype=float)
    if q.shape != (6,):
        raise ValueError("--q must contain exactly 6 values.")
    if degrees:
        q = np.deg2rad(q)
    return q


def _make_material() -> rendering.MaterialRecord:
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    return material


def _paint_mesh(mesh: o3d.geometry.TriangleMesh, link_name: str) -> o3d.geometry.TriangleMesh:
    color = LINK_COLORS.get(link_name)
    if color is not None:
        mesh.paint_uniform_color(color)
    return mesh


class JointSliderViewer:
    def __init__(self, initial_q: np.ndarray):
        self.q = np.asarray(initial_q, dtype=float).reshape(6,)
        self.robot = UR5e()
        self.app = gui.Application.instance
        self.window = self.app.create_window(
            "UR5e Joint Slider Viewer",
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        )

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])

        self.panel = gui.Vert(8, gui.Margins(12, 12, 12, 12))
        self.panel.add_child(gui.Label("UR5e Joint Control"))

        self.joint_controls: list[JointControl] = []
        for joint_idx in range(6):
            self.panel.add_child(gui.Label(f"Joint {joint_idx + 1}"))

            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-JOINT_LIMIT, JOINT_LIMIT)
            slider.double_value = float(self.q[joint_idx])

            value_label = gui.Label(self._format_joint_value(self.q[joint_idx]))
            slider.set_on_value_changed(self._make_slider_callback(joint_idx, value_label))

            self.panel.add_child(slider)
            self.panel.add_child(value_label)
            self.joint_controls.append(JointControl(slider=slider, value_label=value_label))

        reset_button = gui.Button("Reset")
        reset_button.set_on_clicked(self._reset_joints)
        self.panel.add_child(reset_button)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        self._add_robot_geometry()
        self._add_world_frame()
        self._update_robot_pose()
        self._setup_camera()

    @staticmethod
    def _format_joint_value(value: float) -> str:
        return f"{value:+.3f} rad  ({np.rad2deg(value):+.1f} deg)"

    def _make_slider_callback(self, joint_idx: int, value_label: gui.Label):
        def on_value_changed(value: float) -> None:
            self.q[joint_idx] = float(value)
            value_label.text = self._format_joint_value(float(value))
            self._update_robot_pose()

        return on_value_changed

    def _reset_joints(self) -> None:
        self.q[:] = 0.0
        for control in self.joint_controls:
            control.slider.double_value = 0.0
            control.value_label.text = self._format_joint_value(0.0)
        self._update_robot_pose()

    def _on_layout(self, layout_context: gui.LayoutContext) -> None:
        content_rect = self.window.content_rect
        _ = layout_context
        panel_width = PANEL_WIDTH
        self.panel.frame = gui.Rect(
            content_rect.x,
            content_rect.y,
            panel_width,
            content_rect.height,
        )
        self.scene_widget.frame = gui.Rect(
            content_rect.x + panel_width,
            content_rect.y,
            content_rect.width - panel_width,
            content_rect.height,
        )

    def _add_robot_geometry(self) -> None:
        material = _make_material()
        for link_name in LINK_ORDER:
            mesh = self.robot.load_mesh(link_name)
            mesh = _paint_mesh(mesh, link_name)
            self.scene_widget.scene.add_geometry(link_name, mesh, material)

    def _add_world_frame(self) -> None:
        material = _make_material()
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        self.scene_widget.scene.add_geometry("world_frame", frame, material)

    def _setup_camera(self) -> None:
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    def _update_robot_pose(self) -> None:
        link_frames = fk_links(list(M_LIST), S, self.q)
        for link_name, transform in zip(LINK_ORDER, link_frames):
            self.scene_widget.scene.set_geometry_transform(link_name, transform)
        self.window.post_redraw()


def main() -> None:
    parser = argparse.ArgumentParser(description="Display the UR5e with 6 joint sliders.")
    parser.add_argument(
        "--q",
        nargs=6,
        type=float,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Initial joint values. Radians by default, degrees with --degrees.",
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Interpret --q values as degrees.",
    )
    args = parser.parse_args()

    initial_q = _parse_q(args.q, degrees=args.degrees)
    app = gui.Application.instance
    app.initialize()
    JointSliderViewer(initial_q=initial_q)
    app.run()


if __name__ == "__main__":
    main()
