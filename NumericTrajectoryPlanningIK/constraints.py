import numpy as np


# =============================================================================
# Workspace Configuration
# =============================================================================

rMax = 900.0
rMin = 250.0
rMinReach = 100.0


# =============================================================================
# Joint Limits
# =============================================================================

def clampQ(q, jointLimits):
    """
    Clamp joint configuration to joint limits.

    Returns
    -------
    qCmd : clamped joint vector
    hit  : True if any limit was violated
    viol : indices of violating joints
    """

    q = np.asarray(q, dtype=float)

    limits = np.asarray(jointLimits, dtype=float)
    lo = limits[:, 0]
    hi = limits[:, 1]

    violMask = (q < lo) | (q > hi)

    qCmd = np.clip(q, lo, hi)

    viol = np.flatnonzero(violMask).tolist()
    hit = bool(viol)

    return qCmd, hit, viol


# =============================================================================
# Workspace Reach Checks
# =============================================================================

def checkReach(targetXYZ, rMax=rMax, rMin=rMin, margin=0.0):
    """
    Check whether a Cartesian target is reachable based on simple
    spherical workspace limits and inner exclusion zones.
    """

    p = np.asarray(targetXYZ, dtype=float).reshape(3,)
    r = np.linalg.norm(p)

    # --- outer reach ---
    if r > (rMax + margin):

        msg = (
            f"   Distance from base: {r:.1f} mm\n"
            f"   Maximum reach:      {rMax:.1f} mm\n"
            f"   Target:             {p.tolist()}\n"
        )

        print(msg)
        raise ValueError("Unreachable Cartesian target.")

    # --- inner reach ---
    if r < rMin:

        msg = (
            f"   Distance from base: {r:.1f} mm\n"
            f"   Minimum reach:      {rMin:.1f} mm\n"
            f"   Target:             {p.tolist()}\n"
        )

        print(msg)
        raise ValueError("Unreachable Cartesian target.")

    # --- axis dead-zone ---
    coord = ["x", "y", "z"]

    for i in range(3):

        if -rMinReach < p[i] < rMinReach:

            msg = (
                f"  Coordinate out of reach: {coord[i]}\n"
                f"   Minimum axis reach: {rMinReach:.1f} mm\n"
                f"   Target value:       {p[i]:.2f}\n"
            )

            print(msg)
            raise ValueError("Unreachable Cartesian target.")


# =============================================================================
# Trajectory Retiming
# =============================================================================

def retimeTrajectoryLimits(
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

        • joint velocity limits
        • TCP linear velocity limits
        • joint acceleration limits

    Parameters
    ----------
    traj : (N,6) joint trajectory in radians
    Tmax : maximum allowed duration
    vmax_joint_rad_s : scalar or (6,) joint velocity limits
    vmax_tcp_units_s : TCP velocity limit
    fk_pos_fn : FK function returning TCP position
    amax_joint_rad_s : scalar or (6,) joint acceleration limits
    accel_iters : relaxation iterations for acceleration enforcement
    """

    traj = np.asarray(traj, dtype=float)

    if traj.ndim != 2 or traj.shape[1] != 6:
        raise ValueError("traj must be shape (N,6) in radians")

    N = traj.shape[0]

    if N < 2:
        return np.array([0.0]), np.array([])

    # =============================================================================
    # Joint velocity limits
    # =============================================================================

    if vmax_joint_rad_s is None:
        vmax_joint_rad_s = np.deg2rad(180.0)

    vmaxJoint = np.asarray(vmax_joint_rad_s, dtype=float)

    if vmaxJoint.ndim == 0:
        vmaxJoint = np.full(6, float(vmaxJoint))

    if vmaxJoint.shape != (6,) or np.any(vmaxJoint <= 0):
        raise ValueError("vmax_joint_rad_s must be scalar or shape (6,) and positive")

    dq = np.diff(traj, axis=0)

    dtJoint = np.max(np.abs(dq) / vmaxJoint[None, :], axis=1)

    # =============================================================================
    # TCP velocity limits
    # =============================================================================

    useTcp = (vmax_tcp_units_s is not None)

    if useTcp and fk_pos_fn is None:
        raise ValueError("fk_pos_fn must be provided if vmax_tcp_units_s is set")

    if useTcp:

        p = np.vstack([fk_pos_fn(traj[k]) for k in range(N)])
        dp = np.diff(p, axis=0)

        dpNorm = np.linalg.norm(dp, axis=1)

        vmaxTcp = float(vmax_tcp_units_s)

        if vmaxTcp <= 0:
            raise ValueError("vmax_tcp_units_s must be positive")

        dtTcp = dpNorm / vmaxTcp

    else:

        dtTcp = np.zeros_like(dtJoint)

    # initial segment times
    dtSeg = np.maximum(dtJoint, dtTcp)

    # =============================================================================
    # Joint acceleration limits
    # =============================================================================

    if amax_joint_rad_s is not None:

        amax = np.asarray(amax_joint_rad_s, dtype=float)

        if amax.ndim == 0:
            amax = np.full(6, float(amax))

        if amax.shape != (6,) or np.any(amax <= 0):
            raise ValueError("amax_joint_rad_s must be scalar or shape (6,) and positive")

        for _ in range(int(accel_iters)):

            v = dq / dtSeg[:, None]

            dv = np.diff(v, axis=0)

            dtMid = 0.5 * (dtSeg[1:] + dtSeg[:-1])

            a = dv / dtMid[:, None]

            ratio = np.abs(a) / amax[None, :]

            worst = np.max(ratio)

            if worst <= 1.0 + 1e-9:
                break

            s = np.sqrt(np.max(ratio, axis=1))
            s = np.maximum(s, 1.0)

            scale = np.ones_like(dtSeg)

            scale[:-1] = np.maximum(scale[:-1], s)
            scale[1:] = np.maximum(scale[1:], s)

            dtSeg *= scale

    # =============================================================================
    # Build timestamps
    # =============================================================================

    t = np.concatenate([[0.0], np.cumsum(dtSeg)])

    Tmin = float(t[-1])

    if Tmin > Tmax + 1e-9:
        raise ValueError(
            f"Speed/accel limits require Tmin={Tmin:.3f}s > Tmax={Tmax:.3f}s"
        )

    return t, dtSeg