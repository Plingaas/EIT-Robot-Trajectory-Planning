import open3d as o3d
import numpy as np

def make_frame(T: np.ndarray, size=0.1) -> o3d.geometry.TriangleMesh:
    """
    Create a coordinate frame and place it using a 4x4 transform
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T)
    return frame