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

# All dynamic and geometric quantities below are stored in SI units:
# length in m, mass in kg, inertia in kg*m^2.

# Dimensions from UR5e CAD 
q1 = np.array([0, 0, 0], dtype=float) * MM_TO_M
q2 = np.array([0, 0, 162.5], dtype=float) * MM_TO_M
q3 = np.array([0, 0, 587.5], dtype=float) * MM_TO_M
q4 = np.array([0, 0, 979.7], dtype=float) * MM_TO_M
q5 = np.array([0, -133.3, 0.0], dtype=float) * MM_TO_M
q6 = np.array([0, -180.12, 1079.4], dtype=float) * MM_TO_M

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

MOUNT_HOME_FRAME = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

BASE_HOME_FRAME = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 162.5 * MM_TO_M],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

SHOULDER_HOME_FRAME = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -137.8 * MM_TO_M],
    [0.0, 0.0, 1.0, 162.5 * MM_TO_M],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

ELBOW_HOME_FRAME = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -6.0 * MM_TO_M],
    [0.0, 0.0, 1.0, 587.5 * MM_TO_M],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

FOREARM_HOME_FRAME = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -133.3 * MM_TO_M],
    [0.0, 0.0, 1.0, 979.7 * MM_TO_M],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

WRIST_HOME_FRAME = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, -133.3 * MM_TO_M],
    [0.0, 0.0, 1.0, 1079.4 * MM_TO_M],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

END_EFFECTOR_HOME_FRAME = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, -232.9 * MM_TO_M],
    [0.0, 0.0, 1.0, 1079.4 * MM_TO_M],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

HOME_FRAMES = {
    "mount": MOUNT_HOME_FRAME,
    "base": BASE_HOME_FRAME,
    "shoulder": SHOULDER_HOME_FRAME,
    "elbow": ELBOW_HOME_FRAME,
    "forearm": FOREARM_HOME_FRAME,
    "wrist": WRIST_HOME_FRAME,
    "end_effector": END_EFFECTOR_HOME_FRAME,
}

M_LIST = (
    MOUNT_HOME_FRAME,
    BASE_HOME_FRAME,
    SHOULDER_HOME_FRAME,
    ELBOW_HOME_FRAME,
    FOREARM_HOME_FRAME,
    WRIST_HOME_FRAME,
    END_EFFECTOR_HOME_FRAME,
)

# Masses converted from grams to kilograms
MOUNT_MASS = 1392.0 * G_TO_KG
BASE_MASS = 3761.0 * G_TO_KG
SHOULDER_MASS = 8060.0 * G_TO_KG
ELBOW_MASS = 2846.0 * G_TO_KG
FOREARM_MASS = 1373.0 * G_TO_KG
WRIST_MASS = 1300.0 * G_TO_KG
END_EFFECTOR_MASS = 365.0 * G_TO_KG

# Centers of mass converted from millimeters to meters
MOUNT_COM = np.array([0.0, 23.42, 57.36], dtype=float) * MM_TO_M
BASE_COM = np.array([0.0, -10.91, 0.12], dtype=float) * MM_TO_M
SHOULDER_COM = np.array([0.0, -0.08, 212.5], dtype=float) * MM_TO_M
ELBOW_COM = np.array([0.0, -7.29, 168.93], dtype=float) * MM_TO_M
FOREARM_COM = np.array([0.0, 18.27, -2.44], dtype=float) * MM_TO_M
WRIST_COM = np.array([-3.95, 0.0, -7.5], dtype=float) * MM_TO_M
END_EFFECTOR_COM = np.array([-28.5, 0.0, 0.0], dtype=float) * MM_TO_M

# Inertias converted from g*mm^2 to kg*m^2
MOUNT_INERTIA = np.array([
    [1.582e7,  185.81,   124.937],
    [185.81,   7.315e6,  3.177e5],
    [124.937,  3.177e5,  1.155e7]
], dtype=float) * GMM2_TO_KGM2

BASE_INERTIA = np.array([
    [9.629e6,  1074.528,   923.88],
    [1074.528, 8.070e6,  -2.262e5],
    [923.88,  -2.262e5,  8.272e6]
], dtype=float) * GMM2_TO_KGM2

SHOULDER_INERTIA = np.array([
    [6.134e8,   -0.084,     10015.044],
    [-0.084,    6.115e8,   -1.376e5],
    [10015.044, -1.376e5,   1.410e7]
], dtype=float) * GMM2_TO_KGM2

ELBOW_INERTIA = np.array([
    [1.448e8,   -123.568,   -2131.273],
    [-123.568,  1.440e8,     5.772e5],
    [-2131.273, 5.772e5,     3.616e6]
], dtype=float) * GMM2_TO_KGM2

FOREARM_INERTIA = np.array([
    [2.752e6,   -209.815,   -251.176],
    [-209.815,  1.448e6,  -11034.241],
    [-251.176, -11034.241,  2.305e6]
], dtype=float) * GMM2_TO_KGM2

WRIST_INERTIA = np.array([
    [1.258e6,   310.593,    12738.667],
    [310.593,   1.792e6,   -432.147],
    [12738.667, -432.147,   1.486e6]
], dtype=float) * GMM2_TO_KGM2

END_EFFECTOR_INERTIA = np.array([
    [2.533e5,  -0.114,   -229.651],
    [-0.114,   5.024e5,   0.134],
    [-229.651, 0.134,     5.023e5]
], dtype=float) * GMM2_TO_KGM2

G_MOUNT = spatial_inertia(MOUNT_MASS, MOUNT_COM, MOUNT_INERTIA) # Dont think I need this
G_BASE = spatial_inertia(BASE_MASS, BASE_COM, BASE_INERTIA)
G_SHOULDER = spatial_inertia(SHOULDER_MASS, SHOULDER_COM, SHOULDER_INERTIA)
G_ELBOW = spatial_inertia(ELBOW_MASS, ELBOW_COM, ELBOW_INERTIA)
G_FOREARM = spatial_inertia(FOREARM_MASS, FOREARM_COM, FOREARM_INERTIA)
G_WRIST = spatial_inertia(WRIST_MASS, WRIST_COM, WRIST_INERTIA)
G_END_EFFECTOR = spatial_inertia(END_EFFECTOR_MASS, END_EFFECTOR_COM, END_EFFECTOR_INERTIA)

G_LIST = (G_BASE, G_SHOULDER, G_ELBOW, G_FOREARM, G_WRIST, G_END_EFFECTOR)

