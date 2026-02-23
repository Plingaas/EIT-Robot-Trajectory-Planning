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
    
    if r < (rMin):
        msg = (
            f"   Distance from base: {r:.1f} mm\n"
            f"   Minimum reach:      {rMin:.1f} mm\n"
            f"   Target:             {p.tolist()}\n"
        )
        print(msg)
        raise ValueError("Unreachable Cartesian target.")
    
    for i in range(0,3):
        if(p[i]) < rMinReach:

            coord = ["x","y","z"]
            
            msg = (
                f"  Coordinate out of reach: " + coord[i] + " mm\n"
                f"   Minimum reach:      {rMinReach:.1f} mm\n"
                f"   Target:             {p[i]}\n"
                    )
            print(msg)
            raise ValueError("Unreachable Cartesian target.")

