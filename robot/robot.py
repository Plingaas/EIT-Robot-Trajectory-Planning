
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import open3d as o3d
import numpy as np

from core.types import Vector6
from utils.transformation_helper import translate, rotate, assemble_T

@dataclass(frozen=True)
class JointLimit:
    min_pos: np.float64
    max_pos: np.float64
    min_vel: np.float64
    max_vel: np.float64
    min_acc: np.float64
    max_acc: np.float64


@dataclass
class Joint():
    name: str
    mesh_path: str
    mesh: o3d.geometry.TriangleMesh = field(init=False)
    limit: Optional[JointLimit] = None
    T_world: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    
    def __post_init__(self) -> None:
        self.T_world = np.asarray(self.T_world, dtype=float)
        if self.T_world.shape != (4, 4):
            raise ValueError(f"{self.name}: T_world must be 4x4")

        m = o3d.io.read_triangle_mesh(self.mesh_path)
        if m.is_empty():
            raise FileNotFoundError(f"{self.name}: failed to load mesh: {self.mesh_path}")
        m.compute_vertex_normals()

        self.mesh = m
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300.0)


    def set_world_transform(self, T_new_world: np.ndarray) -> None:
        """
        Update mesh pose to absolute world transform (applies delta).
        """
        T_new_world = np.asarray(T_new_world, dtype=float)
        if T_new_world.shape != (4, 4):
            raise ValueError(f"{self.name}: T_new_world must be 4x4")

        T_delta = T_new_world @ np.linalg.inv(self.T_world)
        self.mesh.transform(T_delta)
        self.frame.transform(T_delta)
        self.T_world = T_new_world

    def set_color(self, rgb: List[float]) -> None:
        self.mesh.paint_uniform_color(rgb)

class UR5:
    
    # ---- model file paths ----
    base_mount_path = "model/base_mount_fixed.stl"
    base_joint_path = "model/base_joint_fixed.stl"
    shoulder_joint_path = "model/shoulder_joint_fixed.stl"
    elbow_joint_path = "model/elbow_joint_fixed.stl"
    forearm_joint_path = "model/forearm_joint_fixed.stl"
    wrist_joint_path = "model/wrist_joint_fixed.stl"
    end_effector_joint_path = "model/end_effector_joint_fixed.stl"

    def __init__(self):
        self.joints: Dict[str, Joint] = {
            "mount": Joint("mount", 
                           UR5.base_mount_path, 
                           None),

            "base": Joint("base", 
                          UR5.base_joint_path, 
                          JointLimit(-2*np.pi, 2*np.pi, -np.pi, np.pi, -np.pi, np.pi)),

            "shoulder": Joint("shoulder", 
                              UR5.shoulder_joint_path, 
                          JointLimit(-2*np.pi, 2*np.pi, -np.pi, np.pi, -np.pi, np.pi)),


            "elbow": Joint("elbow", 
                           UR5.elbow_joint_path, 
                          JointLimit(-2*np.pi, 2*np.pi, -np.pi, np.pi, -np.pi, np.pi)),

                           
            "forearm": Joint("forearm", 
                             UR5.forearm_joint_path, 
                          JointLimit(-2*np.pi, 2*np.pi, -np.pi, np.pi, -np.pi, np.pi)),


            "wrist": Joint("wrist", 
                           UR5.wrist_joint_path, 
                          JointLimit(-2*np.pi, 2*np.pi, -np.pi, np.pi, -np.pi, np.pi)),


            "end_effector": Joint("end_effector", 
                                  UR5.end_effector_joint_path, 
                          JointLimit(-2*np.pi, 2*np.pi, -np.pi, np.pi, -np.pi, np.pi)),

        }
    
    
    def meshes(self) -> List[o3d.geometry.TriangleMesh]:
        return [j.mesh for j in self.joints.values()]
    
    def frames(self, size: float = 300.0) -> List[o3d.geometry.TriangleMesh]:
        return [j.frame for j in self.joints.values()]


    def fk_end_effector(self, pose: Vector6) -> np.ndarray:
        
        q1, q2, q3, q4, q5, q6 = pose

        # Forward transform
        R_base_world = rotate(0, 0, q1)
        t_base_world = translate(0, 0, 162.5)
        T_base_world = assemble_T(R_base_world, t_base_world)

        R_shoulder_base = rotate(0, q2, 0)
        t_shoulder_base = translate(0, -137.8, 0)
        T_shoulder_base = assemble_T(R_shoulder_base, t_shoulder_base)
        T_shoulder_world = T_base_world @ T_shoulder_base

        # Forward transform
        R_elbow_shoulder = rotate(0, q3, 0)
        t_elbow_shoulder = translate(0, 131.8, 425)
        T_elbow_shoulder = assemble_T(R_elbow_shoulder, t_elbow_shoulder)
        T_elbow_world = T_shoulder_world @ T_elbow_shoulder

        # Forward transform
        R_forearm_elbow = rotate(0, q4, 0)
        t_forearm_elbow = translate(0, -126.7 - 0.6, 392.2)
        T_forearm_elbow = assemble_T(R_forearm_elbow, t_forearm_elbow)
        T_forearm_world = T_elbow_world @ T_forearm_elbow

        # Forward transform
        R_wrist_forearm = rotate(0, 0, q5-np.pi/2)
        t_wrist_forearm = translate(0, 0, 99.7)
        T_wrist_forearm = assemble_T(R_wrist_forearm, t_wrist_forearm)
        T_wrist_world = T_forearm_world @ T_wrist_forearm

        # Forward transform
        R_end_effector_wrist = rotate(q6, 0, 0)
        t_end_effector_wrist = translate(98.9 + 0.7, 0, 0)
        T_end_effector_wrist= assemble_T(R_end_effector_wrist, t_end_effector_wrist)
        T_end_effector_world = T_wrist_world @ T_end_effector_wrist

        return T_end_effector_world
    
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lb, ub) arrays shape (6,) in radians."""
        order = ["base", "shoulder", "elbow", "forearm", "wrist", "end_effector"]
        lb = np.zeros(6, dtype=float)
        ub = np.zeros(6, dtype=float)
        for i, name in enumerate(order):
            lim = self.joints[name].limit
            if lim is None:
                lb[i], ub[i] = -np.inf, np.inf
            else:
                lb[i], ub[i] = float(lim.min_pos), float(lim.max_pos)
        return lb, ub
    
    def set_joint_pose(self, pose: Vector6):

        q1, q2, q3, q4, q5, q6 = pose
        
        # Forward transform
        R_base_world = rotate(0, 0, q1)
        t_base_world = translate(0, 0, 162.5)
        T_base_world = assemble_T(R_base_world, t_base_world)
        self.joints["base"].set_world_transform(T_base_world)

        R_shoulder_base = rotate(0, q2, 0)
        t_shoulder_base = translate(0, -137.8, 0)
        T_shoulder_base = assemble_T(R_shoulder_base, t_shoulder_base)
        T_shoulder_world = T_base_world @ T_shoulder_base
        self.joints["shoulder"].set_world_transform(T_shoulder_world)

        # Forward transform
        R_elbow_shoulder = rotate(0, q3, 0)
        t_elbow_shoulder = translate(0, 131.8, 425)
        T_elbow_shoulder = assemble_T(R_elbow_shoulder, t_elbow_shoulder)
        T_elbow_world = T_shoulder_world @ T_elbow_shoulder
        self.joints["elbow"].set_world_transform(T_elbow_world)

        # Forward transform
        R_forearm_elbow = rotate(0, q4, 0)
        t_forearm_elbow = translate(0, -126.7 - 0.6, 392.2)
        T_forearm_elbow = assemble_T(R_forearm_elbow, t_forearm_elbow)
        T_forearm_world = T_elbow_world @ T_forearm_elbow
        self.joints["forearm"].set_world_transform(T_forearm_world)

        # Forward transform
        R_wrist_forearm = rotate(0, 0, q5-np.pi/2)
        t_wrist_forearm = translate(0, 0, 99.7)
        T_wrist_forearm = assemble_T(R_wrist_forearm, t_wrist_forearm)
        T_wrist_world = T_forearm_world @ T_wrist_forearm
        self.joints["wrist"].set_world_transform(T_wrist_world)

        # Forward transform
        R_end_effector_wrist = rotate(q6, 0, 0)
        t_end_effector_wrist = translate(98.9 + 0.7, 0, 0)
        T_end_effector_wrist= assemble_T(R_end_effector_wrist, t_end_effector_wrist)
        T_end_effector_world = T_wrist_world @ T_end_effector_wrist
        self.joints["end_effector"].set_world_transform(T_end_effector_world)

    def end_effector_position(self) -> np.ndarray:
        
        # Tracks end effector position
        T = self.joints["end_effector"].T_world
        return T[:3, 3].copy()
