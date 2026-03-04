from core.types import Matrix4x4
from visual.viewer import FrameSequence, Viewer

from robot.ur5e_robot import UR5e


class UR5eViewer(Viewer):
    def __init__(
        self,
        robot: UR5e,
        window_name: str = "UR5e Viewer",
        width: int = 1280,
        height: int = 720,
        world_frame_size: float = 100.0,
        show_frames: bool = False,
        frame_size: float = 50.0,
    ):
        super().__init__(
            window_name=window_name,
            width=width,
            height=height,
            world_frame_size=world_frame_size,
        )
        self.robot = robot
        self.add_robot(self.robot, show_frames=show_frames, frame_size=frame_size)

    def set_link_frames(self, frames: dict[str, Matrix4x4]) -> None:
        self.set_transforms(frames)

    def run_frames(
        self,
        frames: list[dict[str, Matrix4x4]],
        fps: float = 60.0,
        loop: bool = False,
    ) -> None:
        self.run(FrameSequence(frames=frames, fps=fps, loop=loop))
