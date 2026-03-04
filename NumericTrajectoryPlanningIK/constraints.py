import numpy as np

def clampQ(q, jointLimits):
    q = np.asarray(q, dtype=float)

    limits = np.asarray(jointLimits, dtype=float)
    lo = limits[:, 0]
    hi = limits[:, 1]

    violMask = (q < lo) | (q > hi)
    qCmd = np.clip(q, lo, hi)

    viol = np.flatnonzero(violMask).tolist()
    hit = bool(viol)

    return qCmd, hit, viol

rMax = 900.0
rMin = 250.0
rMinReach = 100.0

def checkReach(targetXYZ, rMax=rMax, rMin=rMin, margin=0.0):
    p = np.asarray(targetXYZ, dtype=float).reshape(3,)
    r = np.linalg.norm(p)

    if r > (rMax + margin):
        msg = (
            f"   Distance from base: {r:.1f} mm\n"
            f"   Maximum reach:      {rMax:.1f} mm\n"
            f"   Target:             {p.tolist()}\n"
        )
        print(msg)
        raise ValueError("Unreachable Cartesian target.")
    
    if (r < rMin):
        msg = (
            f"   Distance from base: {r:.1f} mm\n"
            f"   Minimum reach:      {rMin:.1f} mm\n"
            f"   Target:             {p.tolist()}\n"
        )
        print(msg)
        raise ValueError("Unreachable Cartesian target.")
    
    for i in range(0,3):
        if(-rMinReach < (p[i]) < rMinReach):

            coord = ["x","y","z"]
            
            msg = (
                f"  Coordinate out of reach: " + coord[i] + " mm\n"
                f"   Minimum reach:      {rMinReach:.1f} mm\n"
                f"   Target:             {p[i]}\n"
                    )
            print(msg)
            raise ValueError("Unreachable Cartesian target.")

def retime_trajectory_limits(
    traj,
    Tmax,
    vmax_joint_rad_s=None,
    vmax_tcp_units_s=None,
    fk_pos_fn=None,
    amax_joint_rad_s=None,
    accel_iters=30,
):
    """
    Compute timestamps for a joint trajectory so it respects:
      - joint speed limits (rad/s)
      - TCP linear speed limit (units/s), using FK positions
      - joint acceleration limits (rad/s^2)

    traj: (N,6) radians
    Tmax: maximum allowed total duration (s)
    vmax_joint_rad_s: scalar or (6,) array. Default = 180 deg/s.
    vmax_tcp_units_s: scalar. If set, fk_pos_fn must be provided.
    fk_pos_fn: function(q)-> position (3,), e.g. lambda q: forwardKinematicsT(q)[:3,3]
    amax_joint_rad_s: scalar or (6,) array. If None, accel limiting is skipped.
    accel_iters: how many relaxation passes to enforce accel limits.
    """
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 6:
        raise ValueError("traj must be shape (N,6) in radians")
    N = traj.shape[0]
    if N < 2:
        return np.array([0.0]), np.array([])

    # --- defaults from UR manual (velocity) ---
    if vmax_joint_rad_s is None:
        vmax_joint_rad_s = np.deg2rad(180.0)  # 180°/s

    vmax_joint = np.asarray(vmax_joint_rad_s, dtype=float)
    if vmax_joint.ndim == 0:
        vmax_joint = np.full(6, float(vmax_joint))
    if vmax_joint.shape != (6,) or np.any(vmax_joint <= 0):
        raise ValueError("vmax_joint_rad_s must be scalar or shape (6,) and positive")

    use_tcp = (vmax_tcp_units_s is not None)
    if use_tcp and fk_pos_fn is None:
        raise ValueError("fk_pos_fn must be provided if vmax_tcp_units_s is set")

    dq = np.diff(traj, axis=0)  # (N-1,6)

    # --- joint-speed minimum time per segment ---
    dt_joint = np.max(np.abs(dq) / vmax_joint[None, :], axis=1)  # (N-1,)

    # --- TCP-speed minimum time per segment ---
    if use_tcp:
        p = np.vstack([fk_pos_fn(traj[k]) for k in range(N)])  # (N,3)
        dp = np.diff(p, axis=0)                                # (N-1,3)
        dp_norm = np.linalg.norm(dp, axis=1)                   # (N-1,)
        vmax_tcp = float(vmax_tcp_units_s)
        if vmax_tcp <= 0:
            raise ValueError("vmax_tcp_units_s must be positive")
        dt_tcp = dp_norm / vmax_tcp
    else:
        dt_tcp = np.zeros_like(dt_joint)

    # initial segment times satisfy velocity limits
    dt_seg = np.maximum(dt_joint, dt_tcp)

    # --- acceleration limits (optional) ---
    if amax_joint_rad_s is not None:
        amax = np.asarray(amax_joint_rad_s, dtype=float)
        if amax.ndim == 0:
            amax = np.full(6, float(amax))
        if amax.shape != (6,) or np.any(amax <= 0):
            raise ValueError("amax_joint_rad_s must be scalar or shape (6,) and positive")

        # Iteratively increase dt_seg until accel constraints satisfied
        for _ in range(int(accel_iters)):
            v = dq / dt_seg[:, None]  # (N-1,6) segment velocities

            dv = np.diff(v, axis=0)   # (N-2,6)
            dt_mid = 0.5 * (dt_seg[1:] + dt_seg[:-1])  # (N-2,)

            a = dv / dt_mid[:, None]  # (N-2,6)
            ratio = np.abs(a) / amax[None, :]  # (N-2,6)

            worst = np.max(ratio)  # scalar
            if worst <= 1.0 + 1e-9:
                break  # all good

            # scale neighboring segments to reduce acceleration ~ 1/s^2
            s = np.sqrt(np.max(ratio, axis=1))  # (N-2,)
            s = np.maximum(s, 1.0)

            # apply scaling to both adjacent segments for each accel "k"
            # (overlaps are fine; we take multiplicative increases)
            scale = np.ones_like(dt_seg)
            scale[:-1] = np.maximum(scale[:-1], s)   # affects dt_seg[k]
            scale[1:]  = np.maximum(scale[1:],  s)   # affects dt_seg[k+1]

            dt_seg *= scale

    # timestamps
    t = np.concatenate([[0.0], np.cumsum(dt_seg)])
    Tmin = float(t[-1])

    if Tmin > Tmax + 1e-9:
        raise ValueError(f"Speed/accel limits require Tmin={Tmin:.3f}s > Tmax={Tmax:.3f}s")

    return t, dt_seg