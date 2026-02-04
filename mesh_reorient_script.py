import open3d as o3d
import numpy as np
from utils.transformation_helper import rotate, translate, assemble_T

in_path  = "model/end_effector_joint_fixed.stl"
out_path = "model/end_effector_joint.stl"

mesh = o3d.io.read_triangle_mesh(in_path)
if mesh.is_empty():
    raise RuntimeError("Failed to load mesh")

mesh.translate(translate(-53.9, 0, 0))
mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=False)

o3d.visualization.draw_geometries(
    [mesh],
    window_name="UR5 meshes + per-joint frames",
    width=1280,
    height=800,
    mesh_show_back_face=True,
)