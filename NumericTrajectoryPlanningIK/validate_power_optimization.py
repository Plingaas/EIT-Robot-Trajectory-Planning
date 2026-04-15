import matplotlib.pyplot as plt

# =============================================================================
# Data (from your benchmark)
# =============================================================================

waypoints = [10, 15, 25, 50]

runtime = [7.7325, 18.1723, 20.7129, 45.3765]
final_error = [0.116415, 0.073070, 0.043769, 0.016205]
max_joint_step = [0.742490, 0.528002, 0.388604, 0.169950]


# =============================================================================
# Plot 1: Runtime vs Waypoints
# =============================================================================

plt.figure(figsize=(6, 4))

plt.plot(waypoints, runtime, marker='o')
plt.xlabel("Number of Waypoints")
plt.ylabel("Runtime (s)")
plt.title("Runtime vs Trajectory Resolution (FD IK Method)")
plt.grid()

plt.savefig("runtime_plot.png", dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 2: Accuracy + Smoothness
# =============================================================================

plt.figure(figsize=(6, 4))

plt.plot(waypoints, final_error, marker='o', label="Final Pose Error")
plt.plot(waypoints, max_joint_step, marker='o', label="Max Joint Step")

plt.xlabel("Number of Waypoints")
plt.ylabel("Value")
plt.title("Accuracy and Smoothness vs Waypoints")
plt.legend()
plt.grid()

plt.savefig("accuracy_smoothness_plot.png", dpi=300, bbox_inches='tight')
plt.close()


print("Plots generated:")
print(" - runtime_plot.png")
print(" - accuracy_smoothness_plot.png")