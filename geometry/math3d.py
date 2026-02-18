from __future__ import annotations
import math
import numpy as np
import open3d as o3d

from core.types import Vector3

def segment_point_distance(a: Vector3, b: Vector3, p: Vector3) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float); p = np.asarray(p, float)
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / ab2)
    t = max(0.0, min(1.0, t))
    c = a + t * ab
    return float(np.linalg.norm(p - c))

def point_aabb_distance(p: Vector3, lo: Vector3, hi: Vector3) -> float:
    p = np.asarray(p, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    q = np.minimum(np.maximum(p, lo), hi)
    return float(np.linalg.norm(p - q))

def segment_segment_distance(a: Vector3, b: Vector3, c: Vector3, d: Vector3) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = np.asarray(c, float); d = np.asarray(d, float)
    u = b - a; v = d - c; w = a - c
    A = float(np.dot(u, u))
    B = float(np.dot(u, v))
    C = float(np.dot(v, v))
    D = float(np.dot(u, w))
    E = float(np.dot(v, w))
    denom = A * C - B * B

    if denom < 1e-12:
        sc = 0.0
        tc = E / (C + 1e-12)
    else:
        sc = (B * E - C * D) / denom
        tc = (A * E - B * D) / denom

    sc = max(0.0, min(1.0, sc))
    tc = max(0.0, min(1.0, tc))
    p = a + sc * u
    q = c + tc * v
    return float(np.linalg.norm(p - q))

def apply_T_to_vertices(base_V: np.ndarray, T: np.ndarray) -> np.ndarray:
    V = np.asarray(base_V, float)
    T = np.asarray(T, float)
    ones = np.ones((V.shape[0], 1))
    Vh = np.hstack([V, ones])          # Nx4
    Vw = (T @ Vh.T).T[:, :3]           # Nx3
    return Vw

def rodrigues(axis: Vector3, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    axis = axis / n
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)
    I = np.eye(3)
    return I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)

def R_from_z_to_vec(v: Vector3) -> np.ndarray:
    v = np.asarray(v, float)
    v = v / (np.linalg.norm(v) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(np.dot(z, v), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.eye(3)
    if c < -1.0 + 1e-10:
        return rodrigues(np.array([1.0, 0.0, 0.0]), math.pi)
    axis = np.cross(z, v)
    angle = math.acos(c)
    return rodrigues(axis, angle)
