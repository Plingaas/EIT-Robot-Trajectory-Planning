import open3d as o3d

from .robot import UR5

class UR5Viewer:
    def __init__(self, robot: UR5, world_frame_size: float = 300.0):
        self.robot = robot
        self.vis = o3d.visualization.Visualizer()

        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=world_frame_size)

        # IMPORTANT: these must be stable objects (not recreated every call)
        self._meshes = self.robot.meshes()
        self._frames = self.robot.frames()
        self._geoms = [self.world_frame] + self._meshes + self._frames

    def open(self, title="UR5 Viewer", width=1280, height=800):
        self.vis.create_window(title, width=width, height=height)
        for g in self._geoms:
            self.vis.add_geometry(g)

    def tick(self):
        for g in (self._meshes + self._frames):
            self.vis.update_geometry(g)
        self.vis.poll_events()
        self.vis.update_renderer()

    def is_open(self) -> bool:
        return self.vis.poll_events()

    def close(self):
        self.vis.destroy_window()