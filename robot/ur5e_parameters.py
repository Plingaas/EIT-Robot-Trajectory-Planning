import numpy as np

omega_1 = np.array([0, 0, 1], dtype=float)
omega_2 = np.array([0, 1, 0], dtype=float)
omega_3 = np.array([0, 1, 0], dtype=float)
omega_4 = np.array([0, 1, 0], dtype=float)
omega_5 = np.array([0, 0, 1], dtype=float)
omega_6 = np.array([0, 1, 0], dtype=float)

# Dimensions from UR5e CAD, in millimeters
q1 = np.array([0, 0, 0], dtype=float)
q2 = np.array([0, 0, 162.5], dtype=float)
q3 = np.array([0, 0, 587.5], dtype=float)
q4 = np.array([0, 0, 979.7], dtype=float)
q5 = np.array([0, -133.3, 0.0], dtype=float)
q6 = np.array([0, -180.12, 1079.4], dtype=float)

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

r = np.array([
    [0 , 1, 0], 
    [-1, 0, 0], 
    [0 , 0, 1]
], dtype=float)

p = np.array([0, -232.9, 1079.4], dtype=float)

M = np.eye(4, dtype=float)
M[:3, :3] = r
M[:3, 3] = p # This is the home configuration of the end-effector frame in the space frame


# These are inertia in g*mm^2

MOUNT_COM = np.array([0.0, 23.42, 57.36], dtype=float) # This is probably not needed, as its fixed to the ground and wont move
MOUNT_INERTIA = np.array([
    [1.582e7,  185.81,   124.937],
    [185.81,   7.315e6,  3.177e5],
    [124.937,  3.177e5,  1.155e7]
], dtype=float)

BASE_COM = np.array([0.0, -10.91, 0.12], dtype=float)
BASE_INERTIA = np.array([
    [9.629e6,  1074.528,   923.88],
    [1074.528, 8.070e6,  -2.262e5],
    [923.88,  -2.262e5,  8.272e6]
], dtype=float)

SHOULDER_COM = np.array([0.0, -0.08, 212.5], dtype=float)
SHOULDER_INERTIA = np.array([
    [6.134e8,   -0.084,     10015.044],
    [-0.084,    6.115e8,   -1.376e5],
    [10015.044, -1.376e5,   1.410e7]
], dtype=float)

ELBOW_COM = np.array([0.0, -7.29, 168.93], dtype=float)
ELBOW_INERTIA = np.array([
    [1.448e8,   -123.568,   -2131.273],
    [-123.568,  1.440e8,     5.772e5],
    [-2131.273, 5.772e5,     3.616e6]
], dtype=float)

FOREARM_COM = np.array([0.0, 18.27, -2.44], dtype=float)
FOREARM_INERTIA = np.array([
    [2.752e6,   -209.815,   -251.176],
    [-209.815,  1.448e6,  -11034.241],
    [-251.176, -11034.241,  2.305e6]
], dtype=float)

WRIST_COM = np.array([-3.95, 0.0, -7.5], dtype=float)
WRIST_INERTIA = np.array([
    [1.258e6,   310.593,    12738.667],
    [310.593,   1.792e6,   -432.147],
    [12738.667, -432.147,   1.486e6]
], dtype=float)

END_EFFECTOR_COM = np.array([-28.5, 0.0, 0.0], dtype=float)
END_EFFECTOR_INERTIA = np.array([
    [2.533e5,  -0.114,   -229.651],
    [-0.114,   5.024e5,   0.134],
    [-229.651, 0.134,     5.023e5]
], dtype=float)
