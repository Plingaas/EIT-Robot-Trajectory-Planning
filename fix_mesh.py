import open3d as o3d
import numpy as np
from transformation_helper import rotate, translate, assemble_T

in_path  = "model/shoulder_joint.stl"
out_path = "model/shoulder_joint_fixed.stl"

mesh = o3d.io.read_triangle_mesh(in_path)
if mesh.is_empty():
    raise RuntimeError("Failed to load mesh")

mesh.rotate(rotate(-np.pi/2, 0, np.pi))
mesh.translate(translate(0, -414.7, 34))
    

mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=False)

o3d.visualization.draw_geometries(
    [mesh],
    window_name="UR5 meshes + per-joint frames",
    width=1280,
    height=800,
    mesh_show_back_face=True,
)