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
