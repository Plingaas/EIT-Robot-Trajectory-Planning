from transformation_helper import rotate_deg, rotate, translate

import numpy as np

UR5_DH = [
    (0.0,        np.pi/2,  0.089159),   # base -> shoulder
    (-0.42500,  0.0,      0.0),        # shoulder -> elbow
    (-0.39225,  0.0,      0.0),        # elbow -> forearm
    (0.0,        np.pi/2,  0.10915),    # forearm -> wrist1
    (0.0,       -np.pi/2,  0.09465),    # wrist1 -> wrist2
    (0.0,        0.0,      0.0823),     # wrist2 -> tool
]

def dh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d],
        [0,        0,        0,      1],
    ])

T_mount_to_base = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.0895],
    [0, 0, 0, 1]
])