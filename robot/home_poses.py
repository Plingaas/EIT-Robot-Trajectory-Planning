import numpy as np


MESH_PATHS = {
    "mount": "model/base_mount_fixed.stl",
    "base": "model/base_joint_fixed.stl",
    "shoulder": "model/shoulder_joint_fixed.stl",
    "elbow": "model/elbow_joint_fixed.stl",
    "forearm": "model/forearm_joint_fixed.stl",
    "wrist": "model/wrist_joint_fixed.stl",
    "end_effector": "model/end_effector_joint_fixed.stl",
}


HOME_FRAMES = {
    "mount": np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "base": np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 162.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "shoulder": np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -137.8],
            [0.0, 0.0, 1.0, 162.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "elbow": np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -6.0],
            [0.0, 0.0, 1.0, 587.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "forearm": np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -133.3],
            [0.0, 0.0, 1.0, 979.7],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "wrist": np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, -133.3],
            [0.0, 0.0, 1.0, 1079.4],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "end_effector": np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, -232.9],
            [0.0, 0.0, 1.0, 1079.4],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
}
