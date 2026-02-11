import roboticstoolbox as rtb
import numpy as np

# Load the UR5 model
ur5 = rtb.models.UR5()

# Define a goal pose (example end-effector pose as a 4x4 homogeneous transform matrix)
# This is an example, you would replace this with your target T matrix
Tep = ur5.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854]) 

# Solve the IK problem using a numerical solver (e.g., Levenberg-Marquardt)
solution = ur5.ikine_LM(Tep)

# Get the joint angles
if solution.success:
    q = solution.q
    print("Joint angles:", q)
else:
    print("IK solution not found.")
