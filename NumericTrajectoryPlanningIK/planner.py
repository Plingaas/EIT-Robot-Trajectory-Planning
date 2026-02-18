import numpy as np
from kinematics import forwardKinematicsT
from constraints import clampQ

def vectorToR(rv):
    rv = np.asarray(rv, float).reshape(3)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

def RToVector(R):
    R = np.asarray(R, float).reshape(3, 3)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-12:
        return np.zeros(3)
    w = np.array([R[2,1] - R[1,2],
                  R[0,2] - R[2,0],
                  R[1,0] - R[0,1]]) / (2.0 * np.sin(th))
    return th * w

def poseError(q, targets, wPos=1.0, wRot=0.005):
    targets = np.asarray(targets, float).reshape(6)
    T = forwardKinematicsT(q)
    pCur, RCur = T[:3, 3], T[:3, :3]
    pDes = targets[:3]
    RDes = vectorToR(targets[3:])
    ePos = (pDes - pCur)
    eRot = RToVector(RDes @ RCur.T)
    return np.hstack([wPos * ePos, wRot * eRot])

def Jacobianfd(q, targets, eps=1e-6):
    q = np.asarray(q, float)
    n = q.size
    J = np.zeros((6, n))
    e0 = poseError(q, targets)
    for i in range(n):
        dq = np.zeros(n); dq[i] = eps
        J[:, i] = (poseError(q + dq, targets) - e0) / eps
    return J

def solvePoseIK(qInit, targets, jointLimits, qPrev=None,
                maxIters=200, tol=1e-4, damping=1e-3, smoothw=1e-2, stepScale=1.0):
    q = np.asarray(qInit, float).copy()
    qPrev = q.copy() if qPrev is None else np.asarray(qPrev, float).copy()
    n = q.size

    for _ in range(maxIters):
        e = poseError(q, targets)
        if np.linalg.norm(e) < tol:
            break

        J = Jacobianfd(q, targets)

        A = J.T @ J + (damping + smoothw) * np.eye(n)
        b = -(J.T @ e) - smoothw * (q - qPrev)

        q = q + stepScale * np.linalg.solve(A, b)
        q, _, _ = clampQ(q, jointLimits)

    return q

def generateTrajectoryPose(qStart, targets, jointLimits, smoothw=1e-2, **IKKwargs):
    targets = np.asarray(targets, float)
    N = targets.shape[0]
    traj = np.zeros((N, 6))

    q = np.asarray(qStart, float).copy()
    qPrev = q.copy()

    for i in range(N):
        if i == 0 or i == N - 1 or i % max(1, N // 20) == 0:
            pct = 100.0 * i / (N - 1 if N > 1 else 1)
            print(f"\rIK progress: {i+1}/{N} ({pct:5.1f}%)", end="", flush=True)

        q = solvePoseIK(q, targets[i], jointLimits, qPrev=qPrev, smoothw=smoothw, **IKKwargs)
        traj[i] = q
        qPrev = q

    print()
    return traj
