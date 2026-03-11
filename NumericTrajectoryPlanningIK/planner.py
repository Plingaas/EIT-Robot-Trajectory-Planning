import numpy as np
from kinematics import forwardKinematicsT
from constraints import clampQ


# =============================================================================
# Rotation Helpers
# =============================================================================

def vectorToR(rv):
    """
    Convert a 3D rotation vector (axis * angle) into a 3x3 rotation matrix.
    """
    rv = np.asarray(rv, dtype=float).reshape(3)
    theta = np.linalg.norm(rv)

    if theta < 1e-12:
        return np.eye(3)

    k = rv / theta

    K = np.array([
        [0.0, -k[2],  k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])

    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rToVector(R):
    """
    Convert a 3x3 rotation matrix into a rotation vector.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)

    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(c)

    if theta < 1e-10:
        return np.zeros(3)

    if np.pi - theta > 1e-6:
        w = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2.0 * np.sin(theta))

        return theta * w

    A = (R + np.eye(3)) / 2.0
    axis = np.zeros(3)

    axis[0] = np.sqrt(max(A[0, 0], 0.0))
    axis[1] = np.sqrt(max(A[1, 1], 0.0))
    axis[2] = np.sqrt(max(A[2, 2], 0.0))

    if axis[0] > 1e-8:
        axis[1] = np.copysign(axis[1], R[0, 1] + R[1, 0])
        axis[2] = np.copysign(axis[2], R[0, 2] + R[2, 0])
    elif axis[1] > 1e-8:
        axis[2] = np.copysign(axis[2], R[1, 2] + R[2, 1])

    n = np.linalg.norm(axis)

    if n < 1e-10:
        return theta * np.array([1.0, 0.0, 0.0])

    axis /= n
    return theta * axis


# =============================================================================
# Pose Error
# =============================================================================

def poseError(q, targets, wPos=1.0, wRot=0.001):
    """
    Compute the weighted 6D pose error for IK.
    """
    targets = np.asarray(targets, dtype=float).reshape(6)

    T = forwardKinematicsT(q)
    pCur = T[:3, 3]
    rCur = T[:3, :3]

    pDes = targets[:3]
    rDes = vectorToR(targets[3:])

    ePos = pDes - pCur
    eRot = rToVector(rDes @ rCur.T)

    return np.hstack([wPos * ePos, wRot * eRot])


# =============================================================================
# Numerical Jacobian
# =============================================================================

def jacobianFd(q, targets, h=1e-5, wPos=1.0, wRot=0.001):
    """
    Compute the weighted pose-error Jacobian using central finite differences.
    """
    q = np.asarray(q, dtype=float)
    n = q.size

    J = np.zeros((6, n))

    for i in range(n):
        step = h * max(1.0, abs(q[i]))

        dq = np.zeros(n)
        dq[i] = step

        ePlus = poseError(q + dq, targets, wPos=wPos, wRot=wRot)
        eMinus = poseError(q - dq, targets, wPos=wPos, wRot=wRot)

        J[:, i] = (ePlus - eMinus) / (2.0 * step)

    return J


# =============================================================================
# IK Step Helpers
# =============================================================================

def dampedSvdStep(J, e, q, qPrev, lam, smoothw):
    """
    Compute one damped least-squares IK step using SVD.
    """
    U, s, Vt = np.linalg.svd(J, full_matrices=False)

    filt = s / (s**2 + lam**2)

    dqTask = -Vt.T @ (filt * (U.T @ e))
    dqSmooth = -smoothw * (q - qPrev)

    dq = dqTask + dqSmooth

    return dq, s


def adaptiveDamping(sigmaMin, baseDamping, sigmaThresh, singGain):
    """
    Compute adaptive damping from the smallest singular value.
    """
    if sigmaMin < sigmaThresh:
        t = 1.0 - (sigmaMin / sigmaThresh)
        return baseDamping + singGain * (t * t)

    return baseDamping


# =============================================================================
# Main IK Solver
# =============================================================================

def solvePoseIk(
    qInit,
    targets,
    jointLimits,
    qPrev=None,
    maxIters=80,
    tol=1e-4,
    damping=1e-3,
    smoothw=1e-2,
    stepScale=1.0,
    singGain=5e-2,
    sigmaThresh=5e-2,
    maxStepRad=0.15,
    fdStep=1e-5,
    wPos=1.0,
    wRot=0.001,
    minStepTol=1e-8,
    minImproveTol=1e-8,
    stallIters=6,
    returnInfo=False,
):
    """
    Solve a single pose IK problem numerically.
    """
    q = np.asarray(qInit, dtype=float).copy()
    qPrev = q.copy() if qPrev is None else np.asarray(qPrev, dtype=float).copy()

    q, _, _ = clampQ(q, jointLimits)

    info = {
        "converged": False,
        "status": "maxIters",
        "iterations": 0,
        "finalError": None,
        "sigmaMin": None,
        "acceptedSteps": 0,
        "rejectedSteps": 0,
    }

    stallCount = 0
    prevErr = np.inf

    for it in range(maxIters):
        e = poseError(q, targets, wPos=wPos, wRot=wRot)
        err = float(np.linalg.norm(e))

        if err < tol:
            info["converged"] = True
            info["status"] = "converged"
            info["iterations"] = it
            info["finalError"] = err
            return (q, info) if returnInfo else q

        J = jacobianFd(q, targets, h=fdStep, wPos=wPos, wRot=wRot)

        sVals = np.linalg.svd(J, compute_uv=False)
        sigmaMin = float(np.min(sVals))
        info["sigmaMin"] = sigmaMin

        lam = adaptiveDamping(
            sigmaMin=sigmaMin,
            baseDamping=damping,
            sigmaThresh=sigmaThresh,
            singGain=singGain,
        )

        accepted = False
        bestQ = q
        bestErr = err
        bestStepNorm = 0.0

        for _retry in range(8):
            dq, s = dampedSvdStep(J, e, q, qPrev, lam=lam, smoothw=smoothw)
            dq *= stepScale

            stepNorm = float(np.linalg.norm(dq))
            if stepNorm > maxStepRad:
                dq *= (maxStepRad / stepNorm)
                stepNorm = maxStepRad

            qTry = q + dq
            qTry, _, _ = clampQ(qTry, jointLimits)

            eTry = poseError(qTry, targets, wPos=wPos, wRot=wRot)
            errTry = float(np.linalg.norm(eTry))

            if errTry < err:
                accepted = True
                bestQ = qTry
                bestErr = errTry
                bestStepNorm = stepNorm
                info["acceptedSteps"] += 1
                break

            lam *= 3.0
            info["rejectedSteps"] += 1

        if not accepted:
            info["iterations"] = it + 1
            info["finalError"] = err
            info["status"] = "stalledRejectedSteps"
            return (q, info) if returnInfo else q

        q = bestQ

        if bestStepNorm < minStepTol:
            info["iterations"] = it + 1
            info["finalError"] = bestErr
            info["status"] = "stalledSmallStep"
            return (q, info) if returnInfo else q

        improve = prevErr - bestErr
        if improve < minImproveTol:
            stallCount += 1
        else:
            stallCount = 0

        if stallCount >= stallIters:
            info["iterations"] = it + 1
            info["finalError"] = bestErr
            info["status"] = "stalledLowImprovement"
            return (q, info) if returnInfo else q

        prevErr = bestErr

    finalE = poseError(q, targets, wPos=wPos, wRot=wRot)

    info["iterations"] = maxIters
    info["finalError"] = float(np.linalg.norm(finalE))
    info["status"] = "maxIters"

    return (q, info) if returnInfo else q


# =============================================================================
# Trajectory Generation
# =============================================================================

def generateTrajectoryPose(qStart, targets, jointLimits, smoothw=1e-2, **ikKwargs):
    """
    Solve IK for a sequence of pose targets and return a joint trajectory.
    """
    targets = np.asarray(targets, dtype=float)

    q = np.asarray(qStart, dtype=float).copy()
    q, _, _ = clampQ(q, jointLimits)

    nTargets = targets.shape[0]
    nJoints = q.size

    traj = np.zeros((nTargets, nJoints))

    qPrev = q.copy()

    fallbackErrThreshold = ikKwargs.pop("fallbackErrThreshold", 5e-3)
    continuityWeight = ikKwargs.pop("continuityWeight", 0.01)

    for i in range(nTargets):
        if i == 0 or i == nTargets - 1 or i % max(1, nTargets // 20) == 0:
            pct = 100.0 * i / (nTargets - 1 if nTargets > 1 else 1)
            print(f"\rIK progress: {i+1}/{nTargets} ({pct:5.1f}%)", end="", flush=True)

        qMain, infoMain = solvePoseIk(
            qPrev,
            targets[i],
            jointLimits,
            qPrev=qPrev,
            smoothw=smoothw,
            returnInfo=True,
            **ikKwargs,
        )

        eMain = poseError(
            qMain,
            targets[i],
            wPos=ikKwargs.get("wPos", 1.0),
            wRot=ikKwargs.get("wRot", 0.001),
        )
        errMain = float(np.linalg.norm(eMain))

        bestQ = qMain
        bestScore = errMain + continuityWeight * np.linalg.norm(qMain - qPrev)

        goodEnough = infoMain["converged"] or (errMain < fallbackErrThreshold)

        if not goodEnough:
            seedOffsets = [
                np.array([+0.05 if j == 0 else 0.0 for j in range(nJoints)]),
                np.array([-0.05 if j == 0 else 0.0 for j in range(nJoints)]),
                np.array([+0.05 if j == 1 else 0.0 for j in range(nJoints)]) if nJoints > 1 else np.zeros(nJoints),
                np.array([-0.05 if j == 1 else 0.0 for j in range(nJoints)]) if nJoints > 1 else np.zeros(nJoints),
            ]

            for off in seedOffsets:
                s = qPrev + off
                s, _, _ = clampQ(s, jointLimits)

                qTry, infoTry = solvePoseIk(
                    s,
                    targets[i],
                    jointLimits,
                    qPrev=qPrev,
                    smoothw=smoothw,
                    returnInfo=True,
                    **ikKwargs,
                )

                eTry = poseError(
                    qTry,
                    targets[i],
                    wPos=ikKwargs.get("wPos", 1.0),
                    wRot=ikKwargs.get("wRot", 0.001),
                )
                errTry = float(np.linalg.norm(eTry))

                score = errTry + continuityWeight * np.linalg.norm(qTry - qPrev)

                if score < bestScore:
                    bestScore = score
                    bestQ = qTry

        q = bestQ
        traj[i] = q
        qPrev = q

    print()
    return traj


# =============================================================================
# Singularity Metric
# =============================================================================

def singularityCost(q, targets, h=1e-5, wPos=1.0, wRot=0.001, eps=1e-8):
    """
    Compute a simple singularity cost from the smallest singular value of the
    weighted numerical Jacobian.
    """
    J = jacobianFd(q, targets, h=h, wPos=wPos, wRot=wRot)
    s = np.linalg.svd(J, compute_uv=False)
    sigmaMin = float(np.min(s))

    return 1.0 / (sigmaMin**2 + eps)