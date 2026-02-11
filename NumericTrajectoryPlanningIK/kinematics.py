import numpy as np

d1 = 0.089 * 1000
a2 = -0.425 * 1000
a3 = -0.392 * 1000
d4 = 0.109 * 1000
d5 = 0.095 * 1000
d6 = 0.082 * 1000

def dh(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ ct, -st*ca,  st*sa, a*ct],
        [ st,  ct*ca, -ct*sa, a*st],
        [  0,     sa,     ca,    d],
        [  0,      0,      0,    1]
    ])

def forwardKinematicsT(q, returnPoints=False):
    T = np.eye(4)
    q = np.asarray(q, dtype=float)

    if returnPoints:
        pts = [T[:3, 3]]

    DHParams = [
        (0,      np.pi/2, d1, q[0]),
        (a2,     0,       0,  q[1]),
        (a3,     0,       0,  q[2]),
        (0,      np.pi/2, d4, q[3]),
        (0,     -np.pi/2, d5, q[4]),
        (0,      0,       d6, q[5])
    ]

    for p in DHParams:
        T = T @ dh(*p)
        if returnPoints:
            pts.append(T[:3, 3])

    if returnPoints:
        return T, np.array(pts)
    return T
