import numpy as np
import open3d as o3d
from robot import UR5, JointPose

def lerp(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    return (1.0 - s) * a + s * b

def main():
    robot = UR5()

    qa = JointPose(0,   0,   0,   0,   0,   0)
    qb = JointPose(
        np.deg2rad(45), 
        np.deg2rad(80),  
        np.deg2rad(60), 
        np.deg2rad(-45),  
        np.deg2rad(30),  
        np.deg2rad(10))  # degrees if your set_joint_pose expects degrees

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window("UR5 joint move A->B", width=1280, height=800)

    # Add once
    vis.add_geometry(world_frame)
    for g in robot.meshes() + robot.frames():
        vis.add_geometry(g)

    # Move in a loop
    a = qa.as_array()
    b = qb.as_array()

    steps = 240  # ~4 seconds at 60 fps
    while vis.poll_events():
        # Forward and back (ping-pong)
        for s in np.linspace(0.0, 1.0, steps):
            if not vis.poll_events():
                break
            q = lerp(a, b, float(s))
            robot.set_joint_pose(JointPose(*q))

            # Update all geoms (meshes + frames)
            for g in robot.meshes() + robot.frames():
                vis.update_geometry(g)

            vis.update_renderer()

        for s in np.linspace(0.0, 1.0, steps):
            if not vis.poll_events():
                break
            q = lerp(b, a, float(s))
            robot.set_joint_pose(JointPose(*q))

            for g in robot.meshes() + robot.frames():
                vis.update_geometry(g)

            vis.update_renderer()

    vis.destroy_window()

if __name__ == "__main__":
    main()
