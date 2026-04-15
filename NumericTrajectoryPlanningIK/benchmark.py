import csv
import importlib.util
import io
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CREATE_JSON_PATH = SCRIPT_DIR / "create-json.py"
OUTPUT_CSV = SCRIPT_DIR / "fd_benchmark_results.csv"


# =============================================================================
# Dynamic loader for create-json.py
# =============================================================================

def load_create_json_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("create_json", CREATE_JSON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Helpers
# =============================================================================

def parse_candidate_solves(stdout_text):
    match = re.search(r"IK candidate solves:\s*(\d+)", stdout_text)
    if match:
        return int(match.group(1))
    return 0


def compute_total_joint_path_length(traj):
    dq = np.diff(traj, axis=0)
    return float(np.sum(np.linalg.norm(dq, axis=1)))


def run_single_case(module, n_waypoints, use_power_optimization, repeat_idx=0):
    module.nWaypoints = int(n_waypoints)
    module.usePowerOptimization = bool(use_power_optimization)

    # keep settings consistent with the global switch
    module.ikSettings["usePowerOptimization"] = bool(use_power_optimization)

    # build a unique output file for each run
    mode_name = "on" if use_power_optimization else "off"
    module.outputJson = SCRIPT_DIR / f"trajectory_{mode_name}_{n_waypoints}_{repeat_idx}.json"

    qStart = module.getStartConfiguration()
    startPose = module.buildStartPose(qStart)
    targets = module.buildTargets(startPose, module.goalPose, module.nWaypoints)

    stdout_buffer = io.StringIO()

    t0 = time.perf_counter()
    with redirect_stdout(stdout_buffer):
        traj = module.planTrajectory(qStart, targets)
    runtime_plan = time.perf_counter() - t0

    captured = stdout_buffer.getvalue()
    candidate_solves = parse_candidate_solves(captured)

    traj = np.array([module.clampQ(q, module.jointLimits)[0] for q in traj])

    fkPoints = module.computeFkPoints(traj)
    directionDiag = module.computeDirectionDiagnostic(fkPoints, startPose, module.goalPose)
    posDiag = module.computePositionDiagnostic(traj, targets)
    jointDiag = module.computeJointStepDiagnostic(traj)

    t1 = time.perf_counter()
    t, dtSeg = module.retimeTrajectory(traj)
    runtime_retime = time.perf_counter() - t1

    t_min = float(t[-1])

    # stretch to full allowed time, same as current create-json.py
    if t[-1] < module.tMax:
        scale = module.tMax / t[-1]
        t = t * scale
        dtSeg = dtSeg * scale

    module.exportTrajectoryJson(traj, t)

    Tend = module.forwardKinematicsT(traj[-1])
    pEnd = Tend[:3, 3]
    final_error_vec = module.goalPose[:3] - pEnd
    final_error_norm = float(np.linalg.norm(final_error_vec))

    total_joint_path = compute_total_joint_path_length(traj)

    return {
        "repeat": repeat_idx,
        "waypoints": int(n_waypoints),
        "power_optimization": bool(use_power_optimization),
        "runtime_plan_s": runtime_plan,
        "runtime_retime_s": runtime_retime,
        "runtime_total_s": runtime_plan + runtime_retime,
        "candidate_solves": int(candidate_solves),
        "worst_direction_segment": int(directionDiag[0]),
        "worst_direction_dot": float(directionDiag[1]),
        "worst_waypoint": int(posDiag[0]),
        "max_position_error": float(posDiag[1]),
        "worst_joint_segment": int(jointDiag[0]),
        "largest_joint_step_rad": float(jointDiag[1]),
        "t_min_s": t_min,
        "t_used_s": float(t[-1]),
        "final_error_norm": final_error_norm,
        "final_error_x": float(final_error_vec[0]),
        "final_error_y": float(final_error_vec[1]),
        "final_error_z": float(final_error_vec[2]),
        "total_joint_path_length": total_joint_path,
        "output_json": str(module.outputJson.name),
    }


def save_results_csv(results, csv_path):
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results):
    print("\n=== Benchmark Summary ===")

    # group by optimization mode and waypoint count
    grouped = {}
    for row in results:
        key = (row["power_optimization"], row["waypoints"])
        grouped.setdefault(key, []).append(row)

    for (opt, wp), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        runtimes = np.array([r["runtime_total_s"] for r in rows], dtype=float)
        candidate_solves = np.array([r["candidate_solves"] for r in rows], dtype=float)
        final_errors = np.array([r["final_error_norm"] for r in rows], dtype=float)
        max_steps = np.array([r["largest_joint_step_rad"] for r in rows], dtype=float)
        joint_lengths = np.array([r["total_joint_path_length"] for r in rows], dtype=float)

        print(
            f"mode={'ON ' if opt else 'OFF'} | waypoints={wp:3d} | "
            f"runtime={runtimes.mean():8.4f}s | "
            f"candidate_solves={candidate_solves.mean():8.1f} | "
            f"final_error={final_errors.mean():.6f} | "
            f"max_step={max_steps.mean():.6f} rad | "
            f"joint_path={joint_lengths.mean():.6f}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    module = load_create_json_module()

    # -------------------------------------------------------------------------
    # Adjust these for your paper
    # -------------------------------------------------------------------------
    waypoint_counts = [10, 15, 25, 50]
    optimization_modes = [False]
    repeats = 3
    # -------------------------------------------------------------------------

    print("Running FD benchmark...")
    print(f"Goal pose: {module.goalPose.tolist()}")
    print(f"tMax: {module.tMax}")
    print(f"Repeats per setting: {repeats}")

    results = []

    for use_opt in optimization_modes:
        for n_wp in waypoint_counts:
            for r in range(repeats):
                print(
                    f"\nRunning case | opt={'ON' if use_opt else 'OFF'} | "
                    f"waypoints={n_wp} | repeat={r + 1}/{repeats}"
                )

                row = run_single_case(
                    module=module,
                    n_waypoints=n_wp,
                    use_power_optimization=use_opt,
                    repeat_idx=r,
                )
                results.append(row)

                print(
                    f"  runtime_total_s      : {row['runtime_total_s']:.4f}"
                )
                print(
                    f"  candidate_solves     : {row['candidate_solves']}"
                )
                print(
                    f"  final_error_norm     : {row['final_error_norm']:.6f}"
                )
                print(
                    f"  largest_joint_step   : {row['largest_joint_step_rad']:.6f} rad"
                )
                print(
                    f"  total_joint_path_len : {row['total_joint_path_length']:.6f}"
                )
                print(
                    f"  t_min_s              : {row['t_min_s']:.4f}"
                )

    save_results_csv(results, OUTPUT_CSV)
    print_summary(results)

    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()