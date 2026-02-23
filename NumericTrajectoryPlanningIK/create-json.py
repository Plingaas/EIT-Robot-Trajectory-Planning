import json
import numpy as np
from pathlib import Path

from planner import generateTrajectoryPose
from constraints import clampQ, checkReach
from kinematics import forwardKinematicsT


def main():
    dt, T = 0.05, 2.0
    N = int(round(T / dt)) + 1

    jointLimits = list(zip(
        np.deg2rad([-180, -180, -180, -180, -180, -360]),
        np.deg2rad([ 180,  180,  180,  180,  180,  360]),
    ))

    qStart = np.deg2rad([0, -90, 90, 120, -90, 0])
    qStart, _, _ = clampQ(qStart, jointLimits)

    start = np.hstack([forwardKinematicsT(qStart)[:3, 3], [0, 0, 0]])
    goal = np.array([500, 550, 350, 0.0, 0.0, 0.0], float)
    targets = np.linspace(start, goal, N)

    checkReach(goal[:3])

    traj = generateTrajectoryPose(
        qStart,
        targets,
        jointLimits,
        smoothw=2e-3,
        maxIters=400,
        tol=1e-3,
        damping=1e-3,
        stepScale=0.5
    )

    traj = np.array([clampQ(q, jointLimits)[0] for q in traj])

    trajDeg = np.rad2deg(traj)

    waypoints = []
    for i, q in enumerate(trajDeg):
        waypoints.append({
            "t": round(i * dt, 4),
            "q": [float(v) for v in q]
        })

    data = {
        "units": "deg",
        "waypoints": waypoints
    }

    Path("trajectory.json").write_text(json.dumps(data, indent=2))
    print("trajectory.json exported successfully")
    Tend = forwardKinematicsT(traj[-1])
    print("EE xyz:", Tend[:3,3])
    print("Goal :", goal[:3])
    print("Error:", goal[:3] - Tend[:3,3])
    print("Norm :", np.linalg.norm(goal[:3] - Tend[:3,3]))


if __name__ == "__main__":
    main()
