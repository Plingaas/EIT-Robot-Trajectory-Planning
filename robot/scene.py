import open3d as o3d
import numpy as np
from typing import List

from geometry import Geometry
from .robot import UR5

class Scene:
    def __init__(self, robot: UR5, obstacles: List[Geometry], world_frame_size: float = 300.0):
        self.robot = robot
        self.vis = o3d.visualization.Visualizer()

        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=world_frame_size)

        # IMPORTANT: these must be stable objects (not recreated every call)
        self._meshes = self.robot.meshes()
        self._collision_meshes = self.robot.collision_meshes()

        self.obstacles = obstacles
        self.obstacle_meshes = [ob.mesh() for ob in obstacles]

        self._frames = []#self.robot.frames()

        self._trace_pts: list[np.ndarray] = []
        self._trace = o3d.geometry.LineSet()
        self._trace.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self._trace.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        self._trace.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        self._geoms = [self.world_frame] \
            + self._meshes \
            + self._collision_meshes \
            + self.obstacle_meshes \
            + self._frames \
            + [self._trace]

    def open(self, title="Scene", width=1280, height=800):
        self.vis.create_window(title, width=width, height=height)
        for g in self._geoms:
            self.vis.add_geometry(g)

    def tick(self):
        for g in (self._meshes + self._collision_meshes + self._frames + [self._trace]):
            self.vis.update_geometry(g)
        self.vis.poll_events()
        self.vis.update_renderer()

    def is_open(self) -> bool:
        return self.vis.poll_events()

    def close(self):
        self.vis.destroy_window()

    def add_trace_point(self, p_world: np.ndarray, max_points: int = 5000):
        p_world = np.asarray(p_world, dtype=float).reshape(3,)
        self._trace_pts.append(p_world)
    
        if len(self._trace_pts) > max_points:
            self._trace_pts = self._trace_pts[-max_points:]
    
        if len(self._trace_pts) < 2:
            return
    
        P = np.vstack(self._trace_pts)
        L = np.column_stack([np.arange(len(P) - 1), np.arange(1, len(P))]).astype(np.int32)
        C = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(L), 1))
    
        self._trace.points = o3d.utility.Vector3dVector(P)
        self._trace.lines = o3d.utility.Vector2iVector(L)
        self._trace.colors = o3d.utility.Vector3dVector(C)

    def check_collisions(self):
        joints = [joint for joint in self.robot.joints.values()]

        for joint in joints:
            if joint.collision_body is None:
                continue
            for obstacle in self.obstacles:
                a, b = joint.collision_body.endpoints_world()
                r = joint.collision_body.r
                collision = obstacle.collides_capsule(a, b, r)
                if collision:
                    joint.collision_mesh.paint_uniform_color([1, 0, 0])
                else:
                    joint.collision_mesh.paint_uniform_color([0, 1, 0])

        return False 