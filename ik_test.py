import json
import numpy as np

from robot.robot import UR5
from ik.providers.ur5 import UR5KinematicsProvider
from ik.solver import solve_ik  # your provider-based solver entrypoint
from utils.transformation_helper import rotate  # must match your FK convention


# -----------------------------
# Trajectory parameters
# -----------------------------
RADIUS_MM = 150.0

X_CONST = 400.0
CENTER_Y = 0.0
CENTER_Z = 600.0

# Orientation along trajectory (radians) - keep constant
RX = 0.0
RY = 0.0
RZ = 0.0

# Timing / sampling
DT = 0.1                 # seconds per waypoint
T_TOTAL = 10.0           # total duration in seconds
N = int(round(T_TOTAL / DT))  # number of waypoints

# IK tolerances
POS_TOL_MM = 0.1
ROT_TOL_DEG = 0.1

# Optional posture preference (tune q2/q3 for "shoulder high, elbow low" in YOUR convention)
Q_PREF_DEG = np.array([0.0, -60.0, 100.0, 0.0, 0.0, 0.0], dtype=float)
Q_PREF = np.deg2rad(Q_PREF_DEG)

# Strongly bias only shoulder+elbow, leave others mostly free
POSTURE_SIGMA = np.array([999.0, 0.25, 0.25, 999.0, 999.0, 999.0], dtype=float)
SMOOTH_SIGMA  = np.array([0.8,   0.25, 0.25, 1.0,   1.0,   1.0], dtype=float)

POSTURE_WEIGHT = 0.5
SMOOTH_WEIGHT = 0.2

MAX_NFEV = 140


def circle_points_yz(radius_mm: float, x_const: float, cy: float, cz: float, n: int):
    """
    Returns Nx3 points on a circle in the YZ plane, at constant x = x_const.
    Parametrization:
        y = cy + r*cos(theta)
        z = cz + r*sin(theta)
        x = x_const
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.zeros((n, 3), dtype=float)
    pts[:, 0] = x_const
    pts[:, 1] = cy + radius_mm * np.cos(thetas)
    pts[:, 2] = cz + radius_mm * np.sin(thetas)
    return pts


def main():
    robot = UR5()
    provider = UR5KinematicsProvider(robot)

    # Start pose (rad). Use something non-singular if you have one.
    q = np.zeros(6, dtype=float)

    # Precompute constant target rotation matrix once
    R_target = rotate(RX, RY, RZ)

    pts = circle_points_yz(RADIUS_MM, X_CONST, CENTER_Y, CENTER_Z, N)

    waypoints = []

    failures = 0
    for i, p in enumerate(pts):
        t = (i + 1) * DT  # match your example: starts at 0.1, 0.2, ...

        # Solve IK with warm-start + branch locking + posture preference
        res = solve_ik(
            provider=provider,
            xyz_mm=p,
            rxyz=np.array([RX, RY, RZ], dtype=float),   # if your solve_ik expects euler
            q0=q,
            q_prev=q,
            q_pref=Q_PREF,
            pos_tol_mm=POS_TOL_MM,
            rot_tol_deg=ROT_TOL_DEG,
            posture_weight=POSTURE_WEIGHT,
            smooth_weight=SMOOTH_WEIGHT,
            posture_sigma=POSTURE_SIGMA,
            smooth_sigma=SMOOTH_SIGMA,
            max_nfev=MAX_NFEV,
        )

        if not res.success:
            failures += 1
            # Still append something so the trajectory length stays consistent.
            # You could also "break" here if you prefer.
            print(f"[WARN] IK failed at i={i}, t={t:.2f}: {res.message}")

        q = res.q

        waypoints.append({
            "t": float(np.round(t, 10)),
            "q": [float(x) for x in np.rad2deg(q)]
        })

    traj = {
        "units": "deg",
        "waypoints": waypoints
    }

    out_path = "trajectory.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"Waypoints: {len(waypoints)}  Failures: {failures}")


if __name__ == "__main__":
    main()
