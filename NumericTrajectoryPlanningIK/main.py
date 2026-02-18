import numpy as np
from planner import generateTrajectoryPose, Jacobianfd
from visualize import animate
from constraints import clampQ
from kinematics import forwardKinematicsT

dt, T = 0.05, 2.0
N = int(round(T / dt)) + 1

jointLimits = list(zip(
    np.deg2rad([-180, -120, -180, -180, -120, -360]),
    np.deg2rad([ 180,  120,  180,  180,  120,  360]),
))

qStart = np.deg2rad([0, -90, 90, 120, -90, 0])
qStart, _, _ = clampQ(qStart, jointLimits)

start = np.hstack([forwardKinematicsT(qStart)[:3, 3], [0.0, 0.0, 0.0]])
goal = np.array([-400, -400, 400, 0.5, 0.5, 0.5], float)
targets = np.linspace(start, goal, N)

traj = generateTrajectoryPose(qStart, targets, jointLimits,
                               smoothw=2e-3, maxIters=400, tol=1e-6,
                               damping=1e-3, stepScale=0.5)

traj = np.array([clampQ(q, jointLimits)[0] for q in traj])

Tend = forwardKinematicsT(traj[-1])
print("EE xyz:", Tend[:3,3])
print("Goal :", goal[:3])
print("err  :", goal[:3] - Tend[:3,3], "norm", np.linalg.norm(goal[:3] - Tend[:3,3]))

J = Jacobianfd(traj[-1], goal)
u, s, vt = np.linalg.svd(J, full_matrices=False)
print("Jacobian singular values:", s)
print("condition number:", s[0] / s[-1])
print(traj)

animate(traj, dt, jointLimits=jointLimits, tableRows=25)
