import json
from pathlib import Path

import numpy as np

from planner import generateTrajectoryPose
from kinematics import forwardKinematicsT
from energy import computeEnergyCost
from constraints import clampQ, checkReach, retimeTrajectoryLimits


# =============================================================================
# Configuration
# =============================================================================

tMax = 5.0
nWaypoints = 15

jointLimits = list(zip(
    np.deg2rad([-360, -360, -360, -360, -360, -360]),
    np.deg2rad([ 360,  360,  360,  360,  360,  360]),
))

qStartDeg = np.array([0, 0, 0, 0, 0, 0], dtype=float)

# ----CHANGE THIS VECTOR TO TRY NEW TARGETS (xAng, yAng, zAng, xDir, yDir, zDir) -------
goalPose = np.array([600, 400, 300, 0.0, 0.5, 1.0], dtype=float)
# -----------------------------------------------------------------------------

usePowerOptimization = True
powerOptimizationWeight = 1e-3
powerOptimizationStepDt = 1.0

ikSettings = dict(
    smoothw=1e-3,
    maxIters=60,
    tol=1e-3,
    damping=1e-3,
    stepScale=0.5,
    wRot=0.001,
    usePowerOptimization=usePowerOptimization,
    energyWeight=powerOptimizationWeight,
    stepDt=powerOptimizationStepDt,
)

vMaxJoint = np.deg2rad(180.0)
aMaxJoint = np.deg2rad(600.0)
vMaxTcp = 1000.0

outputJson = Path("trajectory.json")


# =============================================================================
# Helpers
# =============================================================================

def getStartConfiguration():
    qStart = np.deg2rad(qStartDeg)
    qStart, _, _ = clampQ(qStart, jointLimits)
    return qStart


def buildStartPose(qStart):
    pStart = forwardKinematicsT(qStart)[:3, 3]
    rStart = np.array([0.0, 0.0, 0.0], dtype=float)
    return np.hstack([pStart, rStart])


def buildTargets(startPose, goalPose, nWaypoints):
    return np.linspace(startPose, goalPose, nWaypoints)


def planTrajectory(qStart, targets):
    traj = generateTrajectoryPose(
        qStart,
        targets,
        jointLimits,
        **ikSettings
    )

    traj = np.array([clampQ(q, jointLimits)[0] for q in traj])
    return traj


def computeFkPoints(traj):
    return np.array([forwardKinematicsT(q)[:3, 3] for q in traj])


# =============================================================================
# Diagnostics
# =============================================================================

def computeDirectionDiagnostic(fkPoints, startPose, goalPose):

    desiredDir = goalPose[:3] - startPose[:3]
    desiredDir = desiredDir / np.linalg.norm(desiredDir)

    worstDot = 1.0
    worstSeg = 0

    for i in range(len(fkPoints) - 1):

        dp = fkPoints[i + 1] - fkPoints[i]
        n = np.linalg.norm(dp)

        if n < 1e-12:
            continue

        actualDir = dp / n
        dot = float(np.dot(actualDir, desiredDir))

        if dot < worstDot:
            worstDot = dot
            worstSeg = i

    return worstSeg, worstDot


def computePositionDiagnostic(traj, targets):

    maxErr = 0.0
    worstWp = 0

    for i, q in enumerate(traj):

        pFk = forwardKinematicsT(q)[:3, 3]
        pTarget = targets[i][:3]

        err = float(np.linalg.norm(pTarget - pFk))

        if err > maxErr:
            maxErr = err
            worstWp = i

    return worstWp, maxErr


def computeJointStepDiagnostic(traj):

    dq = np.diff(traj, axis=0)
    dqNorm = np.linalg.norm(dq, axis=1)

    worstSeg = int(np.argmax(dqNorm))

    return worstSeg, dqNorm[worstSeg], np.rad2deg(dq[worstSeg])


def printDiagnostics(directionDiag, posDiag, jointDiag):

    worstSegDir, worstDot = directionDiag
    worstWp, maxErr = posDiag
    worstSegJoint, maxStep, worstDq = jointDiag

    print("\n--- Trajectory Diagnostics ---")

    print("Worst direction segment:", worstSegDir)
    print("Worst direction dot:", worstDot)

    print("Worst waypoint:", worstWp)
    print("Max position error:", maxErr)

    print("Largest joint step [rad]:", maxStep)
    print("Worst segment index:", worstSegJoint)
    print("Worst dq [deg]:", worstDq)


# =============================================================================
# Retiming + Export
# =============================================================================

def retimeTrajectory(traj):

    t, dtSeg = retimeTrajectoryLimits(
        traj,
        Tmax=tMax,
        vmax_joint_rad_s=vMaxJoint,
        vmax_tcp_units_s=vMaxTcp,
        fk_pos_fn=lambda q: forwardKinematicsT(q)[:3, 3],
        amax_joint_rad_s=aMaxJoint,
    )

    return t, dtSeg


def exportTrajectoryJson(traj, t):

    trajDeg = np.rad2deg(traj)

    waypoints = []

    for i, q in enumerate(trajDeg):
        waypoints.append({
            "t": round(float(t[i]), 4),
            "q": [float(v) for v in q]
        })

    data = {
        "units": "deg",
        "waypoints": waypoints
    }

    outputJson.write_text(json.dumps(data, indent=2))

    print(f"{outputJson.name} exported successfully")


def printFinalReport(traj, goalPose):

    Tend = forwardKinematicsT(traj[-1])
    pEnd = Tend[:3, 3]

    error = goalPose[:3] - pEnd

    print("EE xyz:", pEnd)
    print("Goal :", goalPose[:3])
    print("Error:", error)
    print("Norm :", np.linalg.norm(error))


# =============================================================================
# Main
# =============================================================================

def main():

    print(
        f"Power optimization: {'ON' if usePowerOptimization else 'OFF'} "
        f"(weight={powerOptimizationWeight if usePowerOptimization else 0.0}, stepDt={powerOptimizationStepDt})"
    )

    qStart = getStartConfiguration()
    startPose = buildStartPose(qStart)

    checkReach(goalPose[:3])

    targets = buildTargets(startPose, goalPose, nWaypoints)

    traj = planTrajectory(qStart, targets)

    fkPoints = computeFkPoints(traj)

    directionDiag = computeDirectionDiagnostic(fkPoints, startPose, goalPose)
    posDiag = computePositionDiagnostic(traj, targets)
    jointDiag = computeJointStepDiagnostic(traj)

    printDiagnostics(directionDiag, posDiag, jointDiag)

    t, dtSeg = retimeTrajectory(traj)

    # Stretch to use the full allowed time
    if t[-1] < tMax:
        scale = tMax / t[-1]
        t = t * scale
        dtSeg = dtSeg * scale

    print(f"Retime: Tmin={t[-1]:.4f}s <= Tmax={tMax:.4f}s")

    exportTrajectoryJson(traj, t)

    printFinalReport(traj, goalPose)

    computeEnergyCost(str(outputJson))

    print(
        f"Power optimization: {'ON' if usePowerOptimization else 'OFF'} "
        f"(weight={powerOptimizationWeight if usePowerOptimization else 0.0}, stepDt={powerOptimizationStepDt})"
    )


if __name__ == "__main__":
    main()