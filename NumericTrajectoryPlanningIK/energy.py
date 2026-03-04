import json
import numpy as np

def compute_energy_cost(json_file):
    """
    Estimate energy consumption of a trajectory.

    Model:
        E_total = E_motion + E_idle

    where
        E_motion = a * (J_velocity + beta * J_acceleration)
        E_idle   = P_idle * T_total
    """

    # -------------------------
    # 1. Load trajectory
    # -------------------------
    with open(json_file, 'r') as f:
        data = json.load(f)

    waypoints = data["waypoints"]
    units = data["units"]

    N = len(waypoints)

    t = np.array([waypoints[k]["t"] for k in range(N)], dtype=float)
    q = np.array([waypoints[k]["q"] for k in range(N)], dtype=float)

    # -------------------------
    # 2. Convert units
    # -------------------------
    if units.lower() == "deg":
        q = np.deg2rad(q)

    # -------------------------
    # 3. Time differences
    # -------------------------
    dt_seg = np.diff(t)              # (N-1,)
    dq = np.diff(q, axis=0)          # (N-1,6)

    # -------------------------
    # 4. Joint velocities
    # -------------------------
    qdot = dq / dt_seg[:, None]      # (N-1,6)

    # -------------------------
    # 5. Joint accelerations
    # -------------------------
    dt_mid = 0.5 * (dt_seg[1:] + dt_seg[:-1])
    qddot = np.diff(qdot, axis=0) / dt_mid[:, None]

    # -------------------------
    # 6. Joint weights
    # -------------------------
    W = np.array([3, 6, 5, 1, 0.5, 0.2])

    beta = 0.01

    # -------------------------
    # 7. Velocity effort
    # -------------------------
    J_velocity = np.sum((qdot ** 2) * W[None, :] * dt_seg[:, None])

    # -------------------------
    # 8. Acceleration effort
    # -------------------------
    J_acceleration = np.sum((qddot ** 2) * dt_mid[:, None])

    J_motion = J_velocity + beta * J_acceleration

    # -------------------------
    # 9. Convert effort → Joules
    # -------------------------
    a = 4.0

    E_motion = a * J_motion

    # -------------------------
    # 10. Idle power model
    # -------------------------
    P_idle = 120.0

    T_total = t[-1] - t[0]

    E_idle = P_idle * T_total

    # -------------------------
    # 11. Total energy
    # -------------------------
    E_total = E_motion + E_idle

    print("----- Energy Breakdown -----")
    print(f"Motion effort      : {J_motion:.3f}")
    print(f"Motion energy (J)  : {E_motion:.3f}")
    print(f"Idle energy (J)    : {E_idle:.3f}")
    print(f"Total energy (J)   : {E_total:.3f}")

    return E_total