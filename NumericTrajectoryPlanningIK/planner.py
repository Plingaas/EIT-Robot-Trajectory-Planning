import numpy as np
from kinematics import forwardKinematicsT
from constraints import clampQ
from energy import computeStepEnergyCost


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


def dampedSvdStep(U, s, Vt, e, q, qPrev, lam, smoothw):
    filt = s / (s**2 + lam**2)
    dqTask = -Vt.T @ (filt * (U.T @ e))
    dqSmooth = -smoothw * (q - qPrev)
    dq = dqTask + dqSmooth
    return dq



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

        U, sVals, Vt = np.linalg.svd(J, full_matrices=False)
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
            dq = dampedSvdStep(U, sVals, Vt, e, q, qPrev, lam=lam, smoothw=smoothw)
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
# Trajectory Generation Helpers
# =============================================================================


def defaultStepCost(qPrev, qNext, qPrevPrev=None, dt=1.0, coeffs=None):
    return computeStepEnergyCost(
        qPrev,
        qNext,
        dt=dt,
        qPrevPrev=qPrevPrev,
        coeffs=coeffs,
    )



def estimateBaseSegmentTime(qA, qB, vMaxJoint=None, vMaxTcp=None, fkPosFn=None, minStepDt=1e-3):
    qA = np.asarray(qA, dtype=float)
    qB = np.asarray(qB, dtype=float)

    dq = np.abs(qB - qA)

    dtJoint = 0.0
    if vMaxJoint is not None:
        vMaxJoint = np.asarray(vMaxJoint, dtype=float)
        if vMaxJoint.ndim == 0:
            vMaxJoint = np.full(qA.size, float(vMaxJoint))
        vSafe = np.maximum(vMaxJoint, 1e-9)
        dtJoint = float(np.max(dq / vSafe))

    dtTcp = 0.0
    if vMaxTcp is not None and fkPosFn is not None:
        pA = np.asarray(fkPosFn(qA), dtype=float).reshape(3)
        pB = np.asarray(fkPosFn(qB), dtype=float).reshape(3)
        dp = float(np.linalg.norm(pB - pA))
        dtTcp = dp / max(float(vMaxTcp), 1e-9)

    return max(dtJoint, dtTcp, minStepDt)



def estimateStepTime(
    qPrevPrev,
    qPrev,
    qNext,
    vMaxJoint=None,
    aMaxJoint=None,
    vMaxTcp=None,
    fkPosFn=None,
    minStepDt=1e-3,
    accelIters=6,
):
    qPrev = np.asarray(qPrev, dtype=float)
    qNext = np.asarray(qNext, dtype=float)

    dtNext = estimateBaseSegmentTime(
        qPrev,
        qNext,
        vMaxJoint=vMaxJoint,
        vMaxTcp=vMaxTcp,
        fkPosFn=fkPosFn,
        minStepDt=minStepDt,
    )

    if qPrevPrev is None or aMaxJoint is None:
        return dtNext

    qPrevPrev = np.asarray(qPrevPrev, dtype=float)

    dtPrev = estimateBaseSegmentTime(
        qPrevPrev,
        qPrev,
        vMaxJoint=vMaxJoint,
        vMaxTcp=vMaxTcp,
        fkPosFn=fkPosFn,
        minStepDt=minStepDt,
    )

    aMaxJoint = np.asarray(aMaxJoint, dtype=float)
    if aMaxJoint.ndim == 0:
        aMaxJoint = np.full(qPrev.size, float(aMaxJoint))
    aSafe = np.maximum(aMaxJoint, 1e-9)

    dqPrev = qPrev - qPrevPrev
    dqNext = qNext - qPrev

    for _ in range(int(accelIters)):
        vPrev = dqPrev / max(dtPrev, minStepDt)
        vNext = dqNext / max(dtNext, minStepDt)
        dtMid = 0.5 * (dtPrev + dtNext)
        a = np.abs(vNext - vPrev) / max(dtMid, minStepDt)
        ratio = float(np.max(a / aSafe))

        if ratio <= 1.0 + 1e-9:
            break

        dtNext *= np.sqrt(ratio)

    return max(dtNext, minStepDt)



def buildCandidateSeedOffsets(nJoints, seedStepRad, candidateSeedOffsets=None):
    if candidateSeedOffsets is None:
        seedOffsets = [np.zeros(nJoints)]

        if seedStepRad > 0.0:
            for j in range(nJoints):
                offPlus = np.zeros(nJoints)
                offMinus = np.zeros(nJoints)
                offPlus[j] = seedStepRad
                offMinus[j] = -seedStepRad
                seedOffsets.append(offPlus)
                seedOffsets.append(offMinus)

        return seedOffsets

    seedOffsets = [np.asarray(off, dtype=float).reshape(nJoints) for off in candidateSeedOffsets]

    if not any(np.linalg.norm(off) < 1e-12 for off in seedOffsets):
        seedOffsets = [np.zeros(nJoints)] + seedOffsets

    return seedOffsets



def solveMainCandidate(qPrev, targetsI, jointLimits, smoothw, ikSolveKwargs):
    qMain, infoMain = solvePoseIk(
        qPrev,
        targetsI,
        jointLimits,
        qPrev=qPrev,
        smoothw=smoothw,
        returnInfo=True,
        **ikSolveKwargs,
    )

    return qMain, infoMain



def chooseOldStyleCandidate(
    qPrev,
    qPrevPrev,
    targetsI,
    jointLimits,
    smoothw,
    ikSolveKwargs,
    fallbackErrThreshold,
    continuityWeight,
):
    qMain, infoMain = solveMainCandidate(
        qPrev,
        targetsI,
        jointLimits,
        smoothw,
        ikSolveKwargs,
    )

    errMain = float(np.linalg.norm(poseError(
        qMain,
        targetsI,
        wPos=ikSolveKwargs.get("wPos", 1.0),
        wRot=ikSolveKwargs.get("wRot", 0.001),
    )))

    bestQ = qMain

    dqMain = qMain - qPrev
    if qPrevPrev is None:
        accelMain = 0.0
    else:
        dqPrev = qPrev - qPrevPrev
        accelMain = np.linalg.norm(dqMain - dqPrev)

    accelWeight = ikSolveKwargs.get("accelWeight", 0.2)
    bestScore = (
        errMain
        + continuityWeight * np.linalg.norm(dqMain)
        + accelWeight * accelMain
    )

    solveCount = 1

    goodEnough = infoMain["converged"] or (errMain < fallbackErrThreshold)

    if goodEnough:
        return bestQ, solveCount, False, 0.0, 0.0

    nJoints = qPrev.size
    fallbackOffsets = []

    for j in range(nJoints):
        offPlus = np.zeros(nJoints)
        offMinus = np.zeros(nJoints)
        offPlus[j] = 0.10
        offMinus[j] = -0.10
        fallbackOffsets.append(offPlus)
        fallbackOffsets.append(offMinus)

    for off in fallbackOffsets:
        s = qPrev + off
        s, _, _ = clampQ(s, jointLimits)

        qTry, _infoTry = solvePoseIk(
            s,
            targetsI,
            jointLimits,
            qPrev=qPrev,
            smoothw=smoothw,
            returnInfo=True,
            **ikSolveKwargs,
        )
        solveCount += 1

        errTry = float(np.linalg.norm(poseError(
            qTry,
            targetsI,
            wPos=ikSolveKwargs.get("wPos", 1.0),
            wRot=ikSolveKwargs.get("wRot", 0.001),
        )))
        dqTry = qTry - qPrev
        if qPrevPrev is None:
            accelTry = 0.0
        else:
            dqPrev = qPrev - qPrevPrev
            accelTry = np.linalg.norm(dqTry - dqPrev)
        
        accelWeight = ikSolveKwargs.get("accelWeight", 0.2)
        scoreTry = (
            errTry
            + continuityWeight * np.linalg.norm(dqTry)
            + accelWeight * accelTry
        )

        if scoreTry < bestScore:
            bestScore = scoreTry
            bestQ = qTry

    return bestQ, solveCount, False, 0.0, 0.0



def chooseEnergyOptimalCandidate(
    qPrev,
    qPrevPrev,
    targetsI,
    jointLimits,
    smoothw,
    ikSolveKwargs,
    fallbackErrThreshold,
    continuityWeight,
    energyWeight,
    stepCostFn,
    energyCoeffs,
    candidateSeedOffsets,
    vMaxJoint,
    aMaxJoint,
    vMaxTcp,
    fkPosFn,
    minStepDt,
    nonMainBias,
    minEnergyImprovement,
):
    candidates = []
    solveCount = 0

    for seedIndex, off in enumerate(candidateSeedOffsets):
        s = qPrev + off
        s, _, _ = clampQ(s, jointLimits)

        qTry, infoTry = solvePoseIk(
            s,
            targetsI,
            jointLimits,
            qPrev=qPrev,
            smoothw=smoothw,
            returnInfo=True,
            **ikSolveKwargs,
        )
        solveCount += 1

        errTry = float(np.linalg.norm(poseError(
            qTry,
            targetsI,
            wPos=ikSolveKwargs.get("wPos", 1.0),
            wRot=ikSolveKwargs.get("wRot", 0.001),
        )))

        dtEst = estimateStepTime(
            qPrevPrev,
            qPrev,
            qTry,
            vMaxJoint=vMaxJoint,
            aMaxJoint=aMaxJoint,
            vMaxTcp=vMaxTcp,
            fkPosFn=fkPosFn,
            minStepDt=minStepDt,
        )

        rawEnergy = 0.0 if stepCostFn is None else stepCostFn(
            qPrev,
            qTry,
            qPrevPrev=qPrevPrev,
            dt=dtEst,
            coeffs=energyCoeffs,
        )

        continuityCost = continuityWeight * np.linalg.norm(qTry - qPrev)
        biasCost = nonMainBias if seedIndex != 0 else 0.0
        totalPlanningCost = energyWeight * rawEnergy + continuityCost + biasCost
        validCandidate = infoTry["converged"] or (errTry < fallbackErrThreshold)

        candidates.append({
            "q": qTry,
            "seedIndex": seedIndex,
            "err": errTry,
            "rawEnergy": rawEnergy,
            "continuityCost": continuityCost,
            "totalPlanningCost": totalPlanningCost,
            "dtEst": dtEst,
            "valid": validCandidate,
        })

    validCandidates = [c for c in candidates if c["valid"]]
    if not validCandidates:
        validCandidates = candidates

    validCandidates.sort(key=lambda c: (c["totalPlanningCost"], c["err"], c["seedIndex"]))
    best = validCandidates[0]

    mainCandidate = None
    for c in validCandidates:
        if c["seedIndex"] == 0:
            mainCandidate = c
            break

    if mainCandidate is not None and best["seedIndex"] != 0:
        energyDelta = mainCandidate["totalPlanningCost"] - best["totalPlanningCost"]
        if energyDelta < minEnergyImprovement:
            best = mainCandidate

    switchedCandidate = (best["seedIndex"] != 0)
    return best["q"], solveCount, switchedCandidate, best["rawEnergy"], best["dtEst"]


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

    usePowerOptimization = ikKwargs.pop("usePowerOptimization", False)
    fallbackErrThreshold = ikKwargs.pop("fallbackErrThreshold", 5e-3)
    continuityWeight = ikKwargs.pop("continuityWeight", 0.01)
    energyWeight = ikKwargs.pop("energyWeight", 1.0)
    ikKwargs.pop("stepDt", None)
    ikKwargs.pop("strictCostOptimization", None)
    stepCostFn = ikKwargs.pop("stepCostFn", defaultStepCost)
    energyCoeffs = ikKwargs.pop("energyCoeffs", None)
    seedStepRad = ikKwargs.pop("seedStepRad", 0.05)
    candidateSeedOffsets = ikKwargs.pop("candidateSeedOffsets", None)
    vMaxJoint = ikKwargs.pop("vMaxJoint", None)
    aMaxJoint = ikKwargs.pop("aMaxJoint", None)
    vMaxTcp = ikKwargs.pop("vMaxTcp", None)
    fkPosFn = ikKwargs.pop("fkPosFn", None)
    minStepDt = ikKwargs.pop("minStepDt", 1e-3)
    nonMainBias = ikKwargs.pop("nonMainBias", 0.0)
    minEnergyImprovement = ikKwargs.pop("minEnergyImprovement", 1e-3)

    if fkPosFn is None:
        fkPosFn = lambda qVal: forwardKinematicsT(qVal)[:3, 3]

    candidateSeedOffsets = buildCandidateSeedOffsets(
        nJoints,
        seedStepRad,
        candidateSeedOffsets=candidateSeedOffsets,
    )

    ikSolveKwargs = dict(ikKwargs)

    solveCount = 0
    switchedCandidateCount = 0
    accumulatedPlannerEnergy = 0.0
    accumulatedEstimatedTime = 0.0

    for i in range(nTargets):
        if i == 0 or i == nTargets - 1 or i % max(1, nTargets // 20) == 0:
            pct = 100.0 * i / (nTargets - 1 if nTargets > 1 else 1)
            print(f"\rIK progress: {i+1}/{nTargets} ({pct:5.1f}%)", end="", flush=True)

        qPrevPrev = traj[i - 2].copy() if i >= 2 else None

        if usePowerOptimization:
            q, solveCountI, switchedCandidate, plannerEnergyI, dtEstI = chooseEnergyOptimalCandidate(
                qPrev,
                qPrevPrev,
                targets[i],
                jointLimits,
                smoothw,
                ikSolveKwargs,
                fallbackErrThreshold,
                continuityWeight,
                energyWeight,
                stepCostFn,
                energyCoeffs,
                candidateSeedOffsets,
                vMaxJoint,
                aMaxJoint,
                vMaxTcp,
                fkPosFn,
                minStepDt,
                nonMainBias,
                minEnergyImprovement,
            )
        else:
            q, solveCountI, switchedCandidate, plannerEnergyI, dtEstI = chooseOldStyleCandidate(
                qPrev,
                qPrevPrev,
                targets[i],
                jointLimits,
                smoothw,
                ikSolveKwargs,
                fallbackErrThreshold,
                continuityWeight,
            )

        solveCount += solveCountI
        if switchedCandidate:
            switchedCandidateCount += 1

        accumulatedPlannerEnergy += plannerEnergyI
        accumulatedEstimatedTime += dtEstI

        traj[i] = q
        qPrev = q

    print()
    if usePowerOptimization:
        print(f"IK candidate solves: {solveCount}")
        print(f"IK non-main selections: {switchedCandidateCount}/{nTargets}")
        print(f"Accumulated planner energy score: {accumulatedPlannerEnergy:.6f}")
        print(f"Accumulated estimated step time: {accumulatedEstimatedTime:.6f}s")

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
