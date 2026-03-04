import json
import numpy as np
from pathlib import Path

from planner import generateTrajectoryPose
from kinematics import forwardKinematicsT
from energy import compute_energy_cost
from constraints import clampQ, checkReach, retime_trajectory_limits

def main():
    T = 4.4          # Tmax only
    N = 60            # geometry resolution (try 60–100)

    jointLimits = list(zip(
        np.deg2rad([-360, -360, -360, -360, -360, -360]),
        np.deg2rad([ 360,  360,  360,  360,  360,  360]),
    ))

    qStart = np.deg2rad([0, -90, 90, 120, -90, 0])
    qStart, _, _ = clampQ(qStart, jointLimits)

    start = np.hstack([forwardKinematicsT(qStart)[:3, 3], [0, 0, 0]])
    goal = np.array([-300, 250, 560, 0.75, 0.75, 0.75], float)
    targets = np.linspace(start, goal, N)

    checkReach(goal[:3])

    traj = generateTrajectoryPose(
        qStart,
        targets,
        jointLimits,
        smoothw=1e-3,
        maxIters=400,
        tol=1e-3,
        damping=1e-3,
        stepScale=0.5
    )

    traj = np.array([clampQ(q, jointLimits)[0] for q in traj])


    vmax_joint = np.deg2rad(180.0)
    
    vmax_tcp = 1000.0

    amax_joint = np.deg2rad(600.0)     # choose a conservative accel limit (300°/s²)
    # (You can tune this; if it’s too strict, increase to 500°/s²)

    t, dt_seg = retime_trajectory_limits(
        traj,
        Tmax=T,
        vmax_joint_rad_s=vmax_joint,
        vmax_tcp_units_s=vmax_tcp,
        fk_pos_fn=lambda q: forwardKinematicsT(q)[:3, 3],
        amax_joint_rad_s=amax_joint,
    )
    print(f"Retime: Tmin={t[-1]:.4f}s <= Tmax={T:.4f}s")    

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

    Path("trajectory.json").write_text(json.dumps(data, indent=2))
    print("trajectory.json exported successfully")
    Tend = forwardKinematicsT(traj[-1])
    print("EE xyz:", Tend[:3,3])
    print("Goal :", goal[:3])
    print("Error:", goal[:3] - Tend[:3,3])
    print("Norm :", np.linalg.norm(goal[:3] - Tend[:3,3]))

    compute_energy_cost("trajectory.json")



if __name__ == "__main__":
    main()
