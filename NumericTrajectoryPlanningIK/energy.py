import json
import numpy as np


# =============================================================================
# Energy Model Configuration
# =============================================================================

jointWeights = np.array([3.0, 6.0, 5.0, 1.0, 0.5, 0.2], dtype=float)

# Fitted coefficients
idlePower = 120.0
cAcceleration = 1.0
cVelocity = 1.0
cHold = 1.0


# =============================================================================
# Helpers
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


def computeTimeDifferences(t):
    return np.diff(t)


def computeJointVelocities(q, dtSeg):
    dq = np.diff(q, axis=0)
    qDot = dq / dtSeg[:, None]
    return qDot


def computeJointAccelerations(qDot, dtSeg):
    if len(dtSeg) < 2:
        return np.zeros((len(dtSeg), qDot.shape[1]))

    dtMid = 0.5 * (dtSeg[:-1] + dtSeg[1:])
    qDDotMid = np.diff(qDot, axis=0) / dtMid[:, None]

    # align back to segments
    qDDot = np.zeros_like(qDot)
    qDDot[0] = qDDotMid[0]
    qDDot[-1] = qDDotMid[-1]
    if len(qDot) > 2:
        qDDot[1:-1] = 0.5 * (qDDotMid[:-1] + qDDotMid[1:])
    return qDDot


# =============================================================================
# Feature Extraction
# =============================================================================

def computeTrajectoryFeatures(jsonFile):
    t, q = loadTrajectory(jsonFile)
    dtSeg = computeTimeDifferences(t)
    qDot = computeJointVelocities(q, dtSeg)
    qDDot = computeJointAccelerations(qDot, dtSeg)

    weightedAbsVel = jointWeights[None, :] * np.abs(qDot)
    weightedVel2 = jointWeights[None, :] * (qDot ** 2)
    weightedAccVel = jointWeights[None, :] * np.abs(qDDot) * np.abs(qDot)

    A = float(np.sum(np.sum(weightedAccVel, axis=1) * dtSeg))
    V = float(np.sum(np.sum(weightedVel2, axis=1) * dtSeg))
    H = float(np.sum(np.sum(weightedAbsVel, axis=1) * dtSeg))
    T = float(t[-1] - t[0])

    return {
        "duration": T,
        "A": A,
        "V": V,
        "H": H,
    }


# =============================================================================
# Energy Components
# =============================================================================

def computePredictedEnergyFromFeatures(features, coeffs=None):
    if coeffs is None:
        coeffs = {
            "idlePower": idlePower,
            "cAcceleration": cAcceleration,
            "cVelocity": cVelocity,
            "cHold": cHold,
        }

    T = features["duration"]
    A = features["A"]
    V = features["V"]
    H = features["H"]

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


def computeEnergyCost(jsonFile, coeffs=None):
    features = computeTrajectoryFeatures(jsonFile)
    energies = computePredictedEnergyFromFeatures(features, coeffs)

    print("----- Energy Breakdown -----")
    print(f"Trajectory duration        : {features['duration']:.3f} s")
    print(f"Acceleration feature A     : {features['A']:.6f}")
    print(f"Velocity feature V         : {features['V']:.6f}")
    print(f"Hold feature H             : {features['H']:.6f}")
    print(f"Motion energy              : {energies['eMotion']:.6f}")
    print(f"Idle energy                : {energies['eIdle']:.6f}")
    print(f"Total energy               : {energies['eTotal']:.6f}")

    return energies["eTotal"]


# =============================================================================
# Calibration
# =============================================================================

def fitModelCoefficients(samples):
    """
    samples: list of dicts like
        {
            "trajectory": "traj1.json",
            "energy_j": 812.3
        }
    """
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