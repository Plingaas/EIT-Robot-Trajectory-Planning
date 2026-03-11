import json
import numpy as np


# =============================================================================
# Energy Model Configuration
# =============================================================================

jointWeights = np.array([3, 6, 5, 1, 0.5, 0.2])

betaAcceleration = 0.01

motionScale = 4.0

idlePower = 120.0


# =============================================================================
# Helpers
# =============================================================================

def loadTrajectory(jsonFile):
    """
    Load trajectory data from JSON file.
    Returns time vector and joint trajectory (rad).
    """

    with open(jsonFile, "r") as f:
        data = json.load(f)

    waypoints = data["waypoints"]
    units = data["units"]

    n = len(waypoints)

    t = np.array([waypoints[k]["t"] for k in range(n)], dtype=float)
    q = np.array([waypoints[k]["q"] for k in range(n)], dtype=float)

    if units.lower() == "deg":
        q = np.deg2rad(q)

    return t, q


def computeTimeDifferences(t):
    """
    Compute time intervals between waypoints.
    """
    return np.diff(t)


def computeJointVelocities(q, dtSeg):
    """
    Compute joint velocities qDot.
    """
    dq = np.diff(q, axis=0)
    qDot = dq / dtSeg[:, None]
    return qDot, dq


def computeJointAccelerations(qDot, dtSeg):
    """
    Compute joint accelerations qDDot.
    """
    dtMid = 0.5 * (dtSeg[1:] + dtSeg[:-1])
    qDDot = np.diff(qDot, axis=0) / dtMid[:, None]

    return qDDot, dtMid


# =============================================================================
# Energy Components
# =============================================================================

def computeVelocityEffort(qDot, dtSeg):
    """
    Compute velocity-based effort cost.
    """

    return np.sum((qDot ** 2) * jointWeights[None, :] * dtSeg[:, None])


def computeAccelerationEffort(qDDot, dtMid):
    """
    Compute acceleration-based effort cost.
    """

    return np.sum((qDDot ** 2) * dtMid[:, None])


def computeMotionEnergy(qDot, qDDot, dtSeg, dtMid):
    """
    Compute motion energy from velocity and acceleration effort.
    """

    jVelocity = computeVelocityEffort(qDot, dtSeg)
    jAcceleration = computeAccelerationEffort(qDDot, dtMid)

    jMotion = jVelocity + betaAcceleration * jAcceleration

    eMotion = motionScale * jMotion

    return jMotion, eMotion


def computeIdleEnergy(t):
    """
    Compute idle energy consumption.
    """

    tTotal = t[-1] - t[0]

    eIdle = idlePower * tTotal

    return tTotal, eIdle


# =============================================================================
# Reporting
# =============================================================================

def printEnergyReport(jMotion, eMotion, eIdle, eTotal):

    print("----- Energy Breakdown -----")
    print(f"Motion effort      : {jMotion:.3f}")
    print(f"Motion energy (J)  : {eMotion:.3f}")
    print(f"Idle energy (J)    : {eIdle:.3f}")
    print(f"Total energy (J)   : {eTotal:.3f}")


# =============================================================================
# Main API
# =============================================================================

def computeEnergyCost(jsonFile):
    """
    Estimate energy consumption of a trajectory.

    Model
    -----
    E_total = E_motion + E_idle

    E_motion = a * (J_velocity + beta * J_acceleration)
    E_idle   = P_idle * T_total
    """

    # Load trajectory
    t, q = loadTrajectory(jsonFile)

    # Time intervals
    dtSeg = computeTimeDifferences(t)

    # Joint velocities
    qDot, dq = computeJointVelocities(q, dtSeg)

    # Joint accelerations
    qDDot, dtMid = computeJointAccelerations(qDot, dtSeg)

    # Motion energy
    jMotion, eMotion = computeMotionEnergy(qDot, qDDot, dtSeg, dtMid)

    # Idle energy
    tTotal, eIdle = computeIdleEnergy(t)

    # Total energy
    eTotal = eMotion + eIdle

    # Report
    printEnergyReport(jMotion, eMotion, eIdle, eTotal)

    return eTotal