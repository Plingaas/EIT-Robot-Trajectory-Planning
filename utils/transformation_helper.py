import numpy as np

def translate_x(x):
    return np.array([x, 0, 0])


def translate_y(y):
    return np.array([0, y, 0])


def translate_z(z):
    return np.array([0, 0, z])

def translate(x, y, z):
    return translate_x(x) + translate_y(y) + translate_z(z)

# Returns rotation matrix around the x-axis. Expects radian input
def rotate_x(rx):
    return np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)],
    ])

# Returns rotation matrix around the y-axis. Expects radian input
def rotate_y(ry):
    return np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

# Returns rotation matrix around the z-axis. Expects radian input
def rotate_z(rz):
    return np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1],
    ])

# Returns rotation matrix for RzRyRx. Expects radian inputs
def rotate(rx, ry, rz):
    return rotate_z(rz) @ rotate_y(ry) @ rotate_x(rx)

def rotate_deg(rx, ry, rz):
    rx = np.deg2rad(rx)
    ry = np.deg2rad(ry)
    rz = np.deg2rad(rz)
    return rotate(rx, ry, rz)

def assemble_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T