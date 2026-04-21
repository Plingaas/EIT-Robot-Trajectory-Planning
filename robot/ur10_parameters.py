import numpy as np

from core.spatial import spatial_inertia

MM_TO_M = 1e-3
G_TO_KG = 1e-3
GMM2_TO_KGM2 = 1e-9

omega_1 = np.array([0, 0, 1], dtype=float)
omega_2 = np.array([0, 1, 0], dtype=float)
omega_3 = np.array([0, 1, 0], dtype=float)
omega_4 = np.array([0, 1, 0], dtype=float)
omega_5 = np.array([0, 0, 1], dtype=float)
omega_6 = np.array([0, 1, 0], dtype=float)

# Dimensions from UR10 CAD 
q1 = np.array([0, 0, 0], dtype=float) * MM_TO_M
q2 = np.array([0, -86.0, 128.0], dtype=float) * MM_TO_M
q3 = np.array([0.0, -107.9, 740.1], dtype=float) * MM_TO_M
q4 = np.array([0.0, -109.9, 1311.7], dtype=float) * MM_TO_M
q5 = np.array([0.0, -163.9, 1373.4], dtype=float) * MM_TO_M
q6 = np.array([0.0, -225.6, 1427.4], dtype=float) * MM_TO_M

v1 = -np.cross(omega_1, q1)
v2 = -np.cross(omega_2, q2)
v3 = -np.cross(omega_3, q3)
v4 = -np.cross(omega_4, q4)
v5 = -np.cross(omega_5, q5)
v6 = -np.cross(omega_6, q6)

s1 = np.hstack((omega_1, v1))
s2 = np.hstack((omega_2, v2))
s3 = np.hstack((omega_3, v3))
s4 = np.hstack((omega_4, v4))
s5 = np.hstack((omega_5, v5))
s6 = np.hstack((omega_6, v6))

S = np.column_stack((s1, s2, s3, s4, s5, s6)) # This is the screw axes matrix in the space frame

M_MOUNT = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_BASE = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_SHOULDER = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_ELBOW = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_FOREARM = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_WRIST = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_END_EFFECTOR = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_LIST = (
    M_MOUNT,
    M_BASE,
    M_SHOULDER,
    M_ELBOW,
    M_FOREARM,
    M_WRIST,
    M_END_EFFECTOR,
)

HOME_FRAMES = {
    "mount": np.eye(4, dtype=float),
    "base": np.eye(4, dtype=float),
    "shoulder": np.eye(4, dtype=float),
    "elbow": np.eye(4, dtype=float),
    "forearm": np.eye(4, dtype=float),
    "wrist": np.eye(4, dtype=float),
    "end_effector": np.eye(4, dtype=float),
}

# Masses converted from grams to kilograms
MOUNT_MASS = 1558.0 * G_TO_KG
BASE_MASS = 7369.0 * G_TO_KG
SHOULDER_MASS = 13051.0 * G_TO_KG
ELBOW_MASS = 3989.0 * G_TO_KG
FOREARM_MASS = 2100.0 * G_TO_KG
WRIST_MASS = 1980.0 * G_TO_KG
END_EFFECTOR_MASS = 615.0 * G_TO_KG


