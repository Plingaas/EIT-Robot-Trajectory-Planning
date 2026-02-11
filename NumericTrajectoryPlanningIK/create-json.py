import json
import numpy as np
from pathlib import Path

from planner import generateTrajectoryPose
from constraints import clampQ
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
    goal = np.array([-0.6, -0.6, -0.6, 0.5, 0.5, 0.5], float)
    targets = np.linspace(start, goal, N)

    traj = generateTrajectoryPose(
        qStart,
        targets,
        jointLimits,
        smoothw=2e-3,
        maxIters=200,
        tol=1e-6,
        damping=1e-4,
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


if __name__ == "__main__":
    main()
