import numpy as np

omega_1 = np.array([0, 0, 1], dtype=float)
omega_2 = np.array([0, 1, 0], dtype=float)
omega_3 = np.array([0, 1, 0], dtype=float)
omega_4 = np.array([0, 1, 0], dtype=float)
omega_5 = np.array([0, 0, 1], dtype=float)
omega_6 = np.array([0, 1, 0], dtype=float)

# Dimensions from UR5e CAD, in meters
q1 = np.array([0, 0, 0], dtype=float)
q2 = np.array([0, 0, 0.1625], dtype=float)
q3 = np.array([0, 0, 0.5875], dtype=float)
q4 = np.array([0, 0, 0.9797], dtype=float)
q5 = np.array([0, -0.1333, 0.], dtype=float)
q6 = np.array([0, -0.18012, 1.0794], dtype=float)

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

S = np.column_stack((s1, s2, s3, s4, s5, s6))

r = np.array([
    [0 , 1, 0], 
    [-1, 0, 0], 
    [0 , 0, 1]
], dtype=float)

p = np.array([0, -0.2329, 1.0794], dtype=float)

M = np.eye(4, dtype=float)
M[:3, :3] = r
M[:3, 3] = p


