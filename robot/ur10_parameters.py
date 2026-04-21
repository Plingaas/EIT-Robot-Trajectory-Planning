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
q3 = np.array([0.6, -107.9, 740.1], dtype=float) * MM_TO_M
q4 = np.array([0.6, -109.9, 1311.7], dtype=float) * MM_TO_M
q5 = np.array([0.6, -163.9, 1373.4], dtype=float) * MM_TO_M
q6 = np.array([0.6, -225.6, 1427.4], dtype=float) * MM_TO_M

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

