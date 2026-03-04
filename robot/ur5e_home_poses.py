import numpy as np

MOUNT_HOME_FRAME =  np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
    ],dtype=float
)
BASE_HOME_FRAME = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 162.5],
    [0.0, 0.0, 0.0, 1.0]
    ],dtype=float
)
SHOULDER_HOME_FRAME = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -137.8],
    [0.0, 0.0, 1.0, 162.5],
    [0.0, 0.0, 0.0, 1.0],
    ],dtype=float
)
ELBOW_HOME_FRAME = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -6.0],
    [0.0, 0.0, 1.0, 587.5],
    [0.0, 0.0, 0.0, 1.0],
    ],dtype=float
)
FOREAM_HOME_FRAME = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -133.3],
    [0.0, 0.0, 1.0, 979.7],
    [0.0, 0.0, 0.0, 1.0],
    ],dtype=float
)
WRIST_HOME_FRAME = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, -133.3],
    [0.0, 0.0, 1.0, 1079.4],
    [0.0, 0.0, 0.0, 1.0],
    ],dtype=float
)
END_EFFECTOR_HOME_FRAME = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, -232.9],
    [0.0, 0.0, 1.0, 1079.4],
    [0.0, 0.0, 0.0, 1.0],
    ],dtype=float
)

HOME_FRAMES = {
    "mount": MOUNT_HOME_FRAME,
    "base": BASE_HOME_FRAME,
    "shoulder": SHOULDER_HOME_FRAME,
    "elbow": ELBOW_HOME_FRAME,
    "forearm": FOREAM_HOME_FRAME,
    "wrist": WRIST_HOME_FRAME,
    "end_effector": END_EFFECTOR_HOME_FRAME
}

MESH_PATHS = {
    "mount": "model/base_mount_fixed.stl",
    "base": "model/base_joint_fixed.stl",
    "shoulder": "model/shoulder_joint_fixed.stl",
    "elbow": "model/elbow_joint_fixed.stl",
    "forearm": "model/forearm_joint_fixed.stl",
    "wrist": "model/wrist_joint_fixed.stl",
    "end_effector": "model/end_effector_joint_fixed.stl",
}

LINK_ORDER = (
    "mount",
    "base",
    "shoulder",
    "elbow",
    "forearm",
    "wrist",
    "end_effector",
)