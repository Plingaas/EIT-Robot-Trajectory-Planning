import json
import sys
from pathlib import Path

import numpy as np

try:
    from ..robot.ur5e_parameters import S, M_LIST, G_LIST
    from ..core.dynamics import inverse_dynamics
except ImportError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from robot.ur5e_parameters import S, M_LIST, G_LIST
    from core.dynamics import inverse_dynamics


# =============================================================================
# Inertia Matrix
# =============================================================================

def inertiaMatrix(q):
    """
    Return the joint-space inertia matrix M(q) using inverse dynamics.

    M[:, i] is the torque response to unit joint acceleration e_i with:
        q_dot = 0
        g = 0
        Ftip = 0
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    n = q.size

    q_dot = np.zeros(n, dtype=float)
    g = np.zeros(3, dtype=float)
    Ftip = np.zeros(6, dtype=float)

    M = np.zeros((n, n), dtype=float)

    for i in range(n):
        q_dot_dot = np.zeros(n, dtype=float)
        q_dot_dot[i] = 1.0

        M[:, i] = inverse_dynamics(
            q=q,
            q_dot=q_dot,
            q_dot_dot=q_dot_dot,
            g=g,
            Ftip=Ftip,
            M_LIST=M_LIST,
            G_LIST=G_LIST,
            S=S,
        )

    return 0.5 * (M + M.T)


# =============================================================================
# Energy Model Configuration
# =============================================================================

idlePower = 1.0
cAcceleration = 1.0
cVelocity = 1.0
cHold = 1.0


def getDefaultEnergyCoefficients():
    return {
        "idlePower": idlePower,
        "cAcceleration": cAcceleration,
        "cVelocity": cVelocity,
        "cHold": cHold,
    }


# =============================================================================
# Trajectory Loading
# =============================================================================

def loadTrajectory(jsonFile):
    with open(jsonFile, "r") as f:
        data = json.load(f)

    waypoints = data["waypoints"]
    units = data.get("units", "deg")

    t = np.array([wp["t"] for wp in waypoints], dtype=float)
    q = np.array([wp["q"] for wp in waypoints], dtype=float)

    if units.lower() == "deg":
        q = np.deg2rad(q)
    elif units.lower() != "rad":
        raise ValueError(f"Unsupported units '{units}'")

    if len(t) < 2:
        raise ValueError("Need at least 2 waypoints.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("Waypoint times must be strictly increasing.")

    return t, q


# =============================================================================
# Kinematics Helpers
# =============================================================================

def computeTimeDifferences(t):
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size < 2:
        return np.array([], dtype=float)
    return np.diff(t)


def computeJointVelocities(q, dtSeg):
    q = np.asarray(q, dtype=float)
    dtSeg = np.asarray(dtSeg, dtype=float).reshape(-1)

    if q.ndim != 2:
        raise ValueError("q must be shape (N, n).")
    if len(dtSeg) != q.shape[0] - 1:
        raise ValueError("dtSeg must have length N-1.")

    dq = np.diff(q, axis=0)
    qDot = dq / dtSeg[:, None]
    return qDot


def computeJointAccelerations(qDot, dtSeg):
    qDot = np.asarray(qDot, dtype=float)
    dtSeg = np.asarray(dtSeg, dtype=float).reshape(-1)

    if qDot.ndim != 2:
        raise ValueError("qDot must be shape (N-1, n).")
    if len(dtSeg) != qDot.shape[0]:
        raise ValueError("dtSeg must have same length as qDot rows.")

    if len(dtSeg) < 2:
        return np.zeros_like(qDot)

    dtMid = 0.5 * (dtSeg[:-1] + dtSeg[1:])
    qDDotMid = np.diff(qDot, axis=0) / dtMid[:, None]

    qDDot = np.zeros_like(qDot)
    qDDot[0] = qDDotMid[0]
    qDDot[-1] = qDDotMid[-1]

    if len(qDot) > 2:
        qDDot[1:-1] = 0.5 * (qDDotMid[:-1] + qDDotMid[1:])

    return qDDot


# =============================================================================
# Feature Extraction
# =============================================================================

def computeStepFeatures(qPrev, qNext, dt=1.0, qPrevPrev=None):
    """
    Local step features for compatibility. The full trajectory scorer below
    should be preferred whenever possible.
    """
    qPrev = np.asarray(qPrev, dtype=float).reshape(-1)
    qNext = np.asarray(qNext, dtype=float).reshape(-1)

    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    dq = qNext - qPrev
    qDot = dq / dt

    qMid = 0.5 * (qPrev + qNext)
    M = inertiaMatrix(qMid)

    V = float(qDot @ M @ qDot * dt)
    H = float(np.linalg.norm(M @ qDot) * dt)

    if qPrevPrev is None:
        A = 0.0
    else:
        qPrevPrev = np.asarray(qPrevPrev, dtype=float).reshape(-1)
        qDotPrev = (qPrev - qPrevPrev) / dt
        qDDot = (qDot - qDotPrev) / dt
        A = float(abs(qDDot @ M @ qDot) * dt)

    return {
        "duration": dt,
        "A": A,
        "V": V,
        "H": H,
    }


def computeTrajectoryFeaturesFromArrays(t, q):
    """
    Compute trajectory features directly from arrays.

    Parameters
    ----------
    t : (N,) timestamps
    q : (N, n) joint trajectory in radians
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float)

    if q.ndim != 2:
        raise ValueError("q must be shape (N, n).")
    if len(t) != q.shape[0]:
        raise ValueError("t and q must have matching first dimension.")
    if len(t) < 2:
        raise ValueError("Need at least 2 waypoints.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("Waypoint times must be strictly increasing.")

    dtSeg = computeTimeDifferences(t)
    qDot = computeJointVelocities(q, dtSeg)
    qDDot = computeJointAccelerations(qDot, dtSeg)

    A = 0.0
    V = 0.0
    H = 0.0

    for i in range(len(dtSeg)):
        dt = float(dtSeg[i])

        q_i = q[i]
        qDot_i = qDot[i]
        M = inertiaMatrix(q_i)

        V += float(qDot_i @ M @ qDot_i * dt)
        H += float(np.linalg.norm(M @ qDot_i) * dt)

        if i > 0:
            qDDot_i = qDDot[i]
            A += float(abs(qDDot_i @ M @ qDot_i) * dt)

    T = float(t[-1] - t[0])

    return {
        "duration": T,
        "A": A,
        "V": V,
        "H": H,
    }


def computeTrajectoryFeatures(jsonFile):
    t, q = loadTrajectory(jsonFile)
    return computeTrajectoryFeaturesFromArrays(t, q)


# =============================================================================
# Energy Computation
# =============================================================================

def computePredictedEnergyFromFeatures(features, coeffs=None):
    if coeffs is None:
        coeffs = getDefaultEnergyCoefficients()

    T = float(features["duration"])
    A = float(features["A"])
    V = float(features["V"])
    H = float(features["H"])

    eIdle = coeffs["idlePower"] * T
    eMotion = (
        coeffs["cAcceleration"] * A
        + coeffs["cVelocity"] * V
        + coeffs["cHold"] * H
    )
    eTotal = eIdle + eMotion

    return {
        "eIdle": eIdle,
        "eMotion": eMotion,
        "eTotal": eTotal,
    }


def computeEnergyFromArrays(t, q, coeffs=None, printBreakdown=False):
    features = computeTrajectoryFeaturesFromArrays(t, q)
    energies = computePredictedEnergyFromFeatures(features, coeffs)

    if printBreakdown:
        print("----- Energy Breakdown -----")
        print(f"Trajectory duration        : {features['duration']:.3f} s")
        print(f"Acceleration feature A     : {features['A']:.6f}")
        print(f"Velocity feature V         : {features['V']:.6f}")
        print(f"Hold feature H             : {features['H']:.6f}")
        print(f"Motion energy              : {energies['eMotion']:.6f}")
        print(f"Idle energy                : {energies['eIdle']:.6f}")
        print(f"Total energy               : {energies['eTotal']:.6f}")

    return energies["eTotal"]


def computeStepEnergyCost(qPrev, qNext, dt=1.0, qPrevPrev=None, coeffs=None):
    """
    Kept for compatibility, but planner should prefer computeEnergyFromArrays()
    on a full or partial trajectory.
    """
    features = computeStepFeatures(qPrev, qNext, dt=dt, qPrevPrev=qPrevPrev)
    energies = computePredictedEnergyFromFeatures(features, coeffs)
    return energies["eTotal"]


def computeEnergyCost(jsonFile, coeffs=None):
    t, q = loadTrajectory(jsonFile)
    return computeEnergyFromArrays(t, q, coeffs=coeffs, printBreakdown=True)


# =============================================================================
# Calibration
# =============================================================================

def fitModelCoefficients(samples):
    X = []
    y = []

    for sample in samples:
        features = computeTrajectoryFeatures(sample["trajectory"])
        X.append([
            features["duration"],
            features["A"],
            features["V"],
            features["H"],
        ])
        y.append(float(sample["energy_j"]))

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    return {
        "idlePower": coeffs[0],
        "cAcceleration": coeffs[1],
        "cVelocity": coeffs[2],
        "cHold": coeffs[3],
    }