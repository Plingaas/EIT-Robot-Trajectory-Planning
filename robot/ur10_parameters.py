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
    [0.0, 0.0, 1.0, 38.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_BASE = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -86.0],
    [0.0, 0.0, 1.0, 128.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_SHOULDER = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -107.9],
    [0.0, 0.0, 1.0, 740.1],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_ELBOW = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -109.9],
    [0.0, 0.0, 1.0, 1311.7],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_FOREARM = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -163.9],
    [0.0, 0.0, 1.0, 1373.4],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_WRIST = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -225.6],
    [0.0, 0.0, 1.0, 1427.4],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

M_END_EFFECTOR = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -256.1],
    [0.0, 0.0, 1.0, 1427.4],
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

MOUNT_COM = np.array([0.0, 0.47, -19.56], dtype=float) * MM_TO_M
BASE_COM = np.array([0.0, 76.70, -4.63], dtype=float) * MM_TO_M
SHOULDER_COM = np.array([0.0, -65.88, -361.87], dtype=float) * MM_TO_M
ELBOW_COM = np.array([0.0, 61.08, -318.49], dtype=float) * MM_TO_M
FOREARM_COM = np.array([0.0, 5.85, -57.67], dtype=float) * MM_TO_M
WRIST_COM = np.array([0.0, 57.68, -5.85], dtype=float) * MM_TO_M
END_EFFECTOR_COM = np.array([0.0, 17.47, -0.96], dtype=float) * MM_TO_M

# Inertias converted from g*mm^2 to kg*m^2
MOUNT_INERTIA = np.array([
    [3.178e6,   -108.123,    384.692],
    [-108.123,  3.117e6,   -13666.484],
    [384.692,  -13666.484,  4.722e6]
]) * GMM2_TO_KGM2

BASE_INERTIA = np.array([
    [7.305e7,   -640.201,    -379.742],
    [-640.201,  2.781e7,   -3.480e6],
    [-379.742, -3.480e6,    6.722e7]
], dtype=float) * GMM2_TO_KGM2

SHOULDER_INERTIA = np.array([
    [2.481e9,   59398.274,   15547.264],
    [59398.274, 2.420e9,     3.077e8],
    [15547.264, 3.077e8,     8.862e7]
], dtype=float) * GMM2_TO_KGM2

ELBOW_INERTIA = np.array([
    [6.000e8,   -1.489e5,    7.671e5],
    [-1.489e5,  5.846e8,   -7.843e7],
    [7.671e5,  -7.843e7,    2.035e7]
], dtype=float) * GMM2_TO_KGM2

FOREARM_INERTIA = np.array([
    [1.055e7,   -7812.173,    67179.60],
    [-7812.173, 1.023e7,     -8.085e5],
    [67179.60, -8.085e5,      2.562e6]
], dtype=float) * GMM2_TO_KGM2

WRIST_INERTIA = np.array([
    [9.950e6,   -74046.008,   6567.867],
    [-74046.008, 2.414e6,    -7.615e5],
    [6567.867,  -7.615e5,     9.648e6]
], dtype=float) * GMM2_TO_KGM2

END_EFFECTOR_INERTIA = np.array([
    [5.047e5,   -6461.856,    354.073],
    [-6461.856, 5.787e5,    -8490.359],
    [354.073,  -8490.359,    5.268e5]
], dtype=float) * GMM2_TO_KGM2

G_MOUNT = spatial_inertia(MOUNT_MASS, MOUNT_COM, MOUNT_INERTIA) # Dont think I need this
G_BASE = spatial_inertia(BASE_MASS, BASE_COM, BASE_INERTIA)
G_SHOULDER = spatial_inertia(SHOULDER_MASS, SHOULDER_COM, SHOULDER_INERTIA)
G_ELBOW = spatial_inertia(ELBOW_MASS, ELBOW_COM, ELBOW_INERTIA)
G_FOREARM = spatial_inertia(FOREARM_MASS, FOREARM_COM, FOREARM_INERTIA)
G_WRIST = spatial_inertia(WRIST_MASS, WRIST_COM, WRIST_INERTIA)
G_END_EFFECTOR = spatial_inertia(END_EFFECTOR_MASS, END_EFFECTOR_COM, END_EFFECTOR_INERTIA)

G_LIST = (G_BASE, G_SHOULDER, G_ELBOW, G_FOREARM, G_WRIST, G_END_EFFECTOR)


