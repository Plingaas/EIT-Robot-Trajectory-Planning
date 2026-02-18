import numpy as np

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],
                     [0,c,-s],
                     [0,s,c]])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],
                     [0,1,0],
                     [-s,0,c]])

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],
                     [s,c,0],
                     [0,0,1]])

def T_from_R_t(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T


def forwardKinematicsT(q, returnPoints=False):
    q1, q2, q3, q4, q5, q6 = q

    T0 = np.eye(4)

    T_base = T_from_R_t(
        Rz(q1),
        np.array([0, 0, 99.1 + 63.4])
    )

    T_shoulder = T_from_R_t(
        Ry(q2),
        np.array([0, -137.8, 0])
    )

    T_elbow = T_from_R_t(
        Ry(q3),
        np.array([0, 131.8, 425])
    )

    T_forearm = T_from_R_t(
        Ry(q4),
        np.array([0, -126.7, 392.2])
    )

    T_wrist = T_from_R_t(
        Rz(q5),
        np.array([0, 0, 99.7])
    )

    T_ee = T_from_R_t(
        Rx(q6),
        np.array([98.9, 0, 0])
    )

    Ts = [T0]
    T = T0
    for Ti in (T_base, T_shoulder, T_elbow, T_forearm, T_wrist, T_ee):
        T = T @ Ti
        Ts.append(T)

    if not returnPoints:
        return Ts[-1]

    pts = np.array([Ti[:3, 3] for Ti in Ts])
    return Ts[-1], pts