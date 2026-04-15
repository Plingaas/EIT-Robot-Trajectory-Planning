import json
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from core.dynamics import inverse_dynamics, precompute_inverse_dynamics_constants
from core.energy import compare_potential_energy
from core.kinematics.fk import fk
from core.types import Matrix4x4, Matrix6x6, Matrix6xn, Vector3, Vector6, Vectorn

POLYNOMIAL_DEGREE = 7
STATIONARY_JOINT_TOL = 1e-9
JOINT3_LIMITS = (-2.6, 2.6)
CONSTRAINT_TOL = 1e-9


@dataclass(frozen=True)
class ShapeTrajectory:
    t: Vectorn
    q: np.ndarray
    q_dot: np.ndarray
    q_dot_dot: np.ndarray


@dataclass(frozen=True)
class EnergyEvaluation:
    energy: float
    t: Vectorn
    q: np.ndarray
    q_dot: np.ndarray
    q_dot_dot: np.ndarray
    tau: np.ndarray
    power: Vectorn


@dataclass(frozen=True)
class ShapeOptimizationResult:
    shape_params: Vectorn
    energy: float
    success: bool
    message: str
    iterations: int
    evaluations: int
    cost_history: Vectorn
    best_cost_history: Vectorn
    evaluation: EnergyEvaluation
    joint_delta: Vectorn | None = None
    direction_branch_bits: tuple[int, ...] | None = None
    standard_branch_energy: float | None = None
    standard_branch_success: bool | None = None
    standard_branch_evaluation: EnergyEvaluation | None = None


@dataclass(frozen=True)
class BranchOptimizationResult:
    energy: float
    shape_params: Vectorn
    success: bool
    message: str
    evaluation: EnergyEvaluation
    joint_delta: Vectorn
    direction_branch_bits: tuple[int, ...]
    iterations: int
    evaluations: int
    cost_history: Vectorn
    min_constraint_margin: float


def _ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def save_joint_trajectory(path: str | Path, evaluation: EnergyEvaluation) -> None:
    path = Path(path)
    data = {
        "units": "rad",
        "waypoints": [
            {
                "t": float(t_k),
                "q": [float(value) for value in q_k],
            }
            for t_k, q_k in zip(evaluation.t, evaluation.q)
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot_joint_trajectory_comparison(
    cubic_evaluation: EnergyEvaluation,
    optimized_evaluation: EnergyEvaluation,
    polynomial_degree: int,
) -> None:
    n_joints = cubic_evaluation.q.shape[1]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    axes = axes.ravel()

    for joint_idx in range(n_joints):
        ax = axes[joint_idx]
        ax.plot(
            cubic_evaluation.t,
            cubic_evaluation.q[:, joint_idx],
            label="Baseline 5th Degree",
            linewidth=2.0,
        )
        ax.plot(
            optimized_evaluation.t,
            optimized_evaluation.q[:, joint_idx],
            label=f"Optimized {_ordinal(polynomial_degree)} Degree",
            linewidth=2.0,
        )
        ax.set_title(f"Joint {joint_idx + 1}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("q [rad]")
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    fig.tight_layout()
    plt.show()


def plot_cost_history(result: ShapeOptimizationResult) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    evaluations = np.arange(1, result.cost_history.size + 1)
    ax.plot(evaluations, result.cost_history, label="Cost", alpha=0.5)
    ax.plot(evaluations, result.best_cost_history, label="Best So Far", linewidth=2.0)
    ax.set_title("Optimization Cost History")
    ax.set_xlabel("Function Evaluation")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def _cumulative_integral(y: Vectorn, x: Vectorn) -> Vectorn:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape.")

    cumulative = np.zeros_like(y, dtype=float)
    if y.size < 2:
        return cumulative

    dx = np.diff(x)
    trapezoids = 0.5 * (y[:-1] + y[1:]) * dx
    cumulative[1:] = np.cumsum(trapezoids)
    return cumulative


def plot_power_comparison(
    cubic_evaluation: EnergyEvaluation,
    optimized_evaluation: EnergyEvaluation,
    polynomial_degree: int,
    standard_branch_evaluation: EnergyEvaluation | None = None,
) -> None:
    cubic_cumulative = _cumulative_integral(cubic_evaluation.power, cubic_evaluation.t)
    optimized_cumulative = _cumulative_integral(optimized_evaluation.power, optimized_evaluation.t)
    standard_cumulative = None
    if standard_branch_evaluation is not None:
        standard_cumulative = _cumulative_integral(
            standard_branch_evaluation.power,
            standard_branch_evaluation.t,
        )

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(
        cubic_evaluation.t,
        cubic_evaluation.power,
        label="Baseline 5th Degree",
        linewidth=2.0,
    )
    axes[0].plot(
        optimized_evaluation.t,
        optimized_evaluation.power,
        label=f"Best {_ordinal(polynomial_degree)} Degree",
        linewidth=2.0,
    )
    if standard_branch_evaluation is not None:
        axes[0].plot(
            standard_branch_evaluation.t,
            standard_branch_evaluation.power,
            label=f"Standard Branch {_ordinal(polynomial_degree)} Degree",
            linewidth=2.0,
            linestyle="--",
        )
    axes[0].set_title("Power Consumption")
    axes[0].set_ylabel("Power")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        cubic_evaluation.t,
        cubic_cumulative,
        label="Baseline 5th Degree",
        linewidth=2.0,
    )
    axes[1].plot(
        optimized_evaluation.t,
        optimized_cumulative,
        label=f"Best {_ordinal(polynomial_degree)} Degree",
        linewidth=2.0,
    )
    if standard_branch_evaluation is not None:
        axes[1].plot(
            standard_branch_evaluation.t,
            standard_cumulative,
            label=f"Standard Branch {_ordinal(polynomial_degree)} Degree",
            linewidth=2.0,
            linestyle="--",
        )
    axes[1].set_title("Cumulative Energy")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Energy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def _integrate_power(power: Vectorn, time: Vectorn) -> float:
    return float(np.trapezoid(power, time))


def _shortest_angular_delta(q_start: Vectorn, q_goal: Vectorn) -> Vectorn:
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    return (q_goal - q_start + np.pi) % (2.0 * np.pi) - np.pi


def _stationary_joint_mask(q_start: Vectorn, q_goal: Vectorn) -> np.ndarray:
    return np.abs(_shortest_angular_delta(q_start, q_goal)) <= STATIONARY_JOINT_TOL


def _stationary_delta_mask(delta: Vectorn) -> np.ndarray:
    return np.abs(np.asarray(delta, dtype=float).reshape(-1)) <= STATIONARY_JOINT_TOL


def _validate_joint3_endpoint(
    q: Vectorn,
    name: str,
    joint3_limits: tuple[float, float] = JOINT3_LIMITS,
) -> None:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size < 3:
        raise ValueError("joint3 limits require at least three joints.")

    lower, upper = joint3_limits
    if lower >= upper:
        raise ValueError("joint3_limits must satisfy lower < upper.")

    q3 = float(q[2])
    if not (lower < q3 < upper):
        raise ValueError(
            f"{name} violates joint 3 limits: {q3:.6f} rad is not in "
            f"({lower:.6f}, {upper:.6f})."
        )


def _joint3_limit_constraint_values(
    q_start: Vectorn,
    q_goal: Vectorn,
    duration: float,
    shape_params: Vectorn,
    polynomial_degree: int,
    num_samples: int,
    freeze_stationary_joints: bool,
    joint_delta: Vectorn | None,
    joint3_limits: tuple[float, float],
) -> Vectorn:
    lower, upper = joint3_limits
    trajectory = sample_shaped_joint_trajectory(
        q_start=q_start,
        q_goal=q_goal,
        duration=duration,
        shape_params=shape_params,
        polynomial_degree=polynomial_degree,
        num_samples=num_samples,
        freeze_stationary_joints=freeze_stationary_joints,
        joint_delta=joint_delta,
    )
    q3 = trajectory.q[:, 2]
    return np.concatenate((q3 - lower, upper - q3))


def _clockwise_counterclockwise_delta_options(q_start: Vectorn, q_goal: Vectorn) -> np.ndarray:
    shortest_delta = _shortest_angular_delta(q_start, q_goal)
    opposite_delta = np.where(
        shortest_delta >= 0.0,
        shortest_delta - 2.0 * np.pi,
        shortest_delta + 2.0 * np.pi,
    )
    return np.column_stack((shortest_delta, opposite_delta))


def _num_shape_coefficients(polynomial_degree: int) -> int:
    if polynomial_degree < 5:
        raise ValueError("polynomial_degree must be at least 5.")
    return polynomial_degree - 5


def _reshape_shape_params(shape_params: Vectorn, n_joints: int, polynomial_degree: int) -> np.ndarray:
    shape_params = np.asarray(shape_params, dtype=float)
    n_coeffs = _num_shape_coefficients(polynomial_degree)
    if shape_params.shape == (n_joints, n_coeffs):
        return shape_params
    if shape_params.shape == (n_coeffs * n_joints,):
        return shape_params.reshape(n_joints, n_coeffs)
    raise ValueError(
        f"shape_params must have shape ({n_coeffs * n_joints},) or ({n_joints}, {n_coeffs})."
    )


def _shape_basis(u: Vectorn, polynomial_degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_values = []
    basis_first = []
    basis_second = []

    for p in range(3, polynomial_degree - 2):
        basis_values.append(u**p - 3.0 * u**(p + 1) + 3.0 * u**(p + 2) - u**(p + 3))
        basis_first.append(
            p * u**(p - 1)
            - 3.0 * (p + 1) * u**p
            + 3.0 * (p + 2) * u**(p + 1)
            - (p + 3) * u**(p + 2)
        )
        basis_second.append(
            p * (p - 1) * u**(p - 2)
            - 3.0 * p * (p + 1) * u**(p - 1)
            + 3.0 * (p + 1) * (p + 2) * u**p
            - (p + 2) * (p + 3) * u**(p + 1)
        )
    if not basis_values:
        zeros = np.zeros((u.size, 0), dtype=float)
        return zeros, zeros, zeros

    return (
        np.column_stack(basis_values),
        np.column_stack(basis_first),
        np.column_stack(basis_second),
    )


def sample_shaped_joint_trajectory(
    q_start: Vectorn,
    q_goal: Vectorn,
    duration: float,
    shape_params: Vectorn,
    polynomial_degree: int = POLYNOMIAL_DEGREE,
    num_samples: int = 200,
    freeze_stationary_joints: bool = False,
    joint_delta: Vectorn | None = None,
) -> ShapeTrajectory:
    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=float).reshape(-1)

    if q_start.shape != q_goal.shape:
        raise ValueError("q_start and q_goal must have the same shape.")
    if duration <= 0.0:
        raise ValueError("duration must be > 0.")
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")

    n_joints = q_start.size
    coeffs = _reshape_shape_params(shape_params, n_joints, polynomial_degree)
    if joint_delta is None:
        dq = _shortest_angular_delta(q_start, q_goal)
    else:
        dq = np.asarray(joint_delta, dtype=float).reshape(-1)
        if dq.shape != q_start.shape:
            raise ValueError("joint_delta must have the same shape as q_start.")

    if freeze_stationary_joints:
        coeffs = coeffs.copy()
        coeffs[_stationary_delta_mask(dq), :] = 0.0

    t = np.linspace(0.0, duration, num_samples, dtype=float)
    u = t / duration

    # Base quintic: fixed endpoints with zero endpoint velocity and acceleration.
    h = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    h_u = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    h_uu = 60.0 * u - 180.0 * u**2 + 120.0 * u**3

    # Boundary-safe shape terms:
    # phi(0)=phi(1)=phi'(0)=phi'(1)=phi''(0)=phi''(1)=0.
    phi, phi_u, phi_uu = _shape_basis(u, polynomial_degree)

    q = (
        q_start[None, :]
        + h[:, None] * dq[None, :]
        + phi @ coeffs.T
    )
    q_dot = (
        h_u[:, None] * dq[None, :]
        + phi_u @ coeffs.T
    ) / duration
    q_dot_dot = (
        h_uu[:, None] * dq[None, :]
        + phi_uu @ coeffs.T
    ) / (duration**2)

    return ShapeTrajectory(t=t, q=q, q_dot=q_dot, q_dot_dot=q_dot_dot)


def evaluate_shaped_trajectory_energy(
    q_start: Vectorn,
    q_goal: Vectorn,
    duration: float,
    shape_params: Vectorn,
    g: Vector3,
    Ftip: Vector6,
    payload_mass: float,
    M_LIST: tuple[Matrix4x4, ...] | list[Matrix4x4],
    G_LIST: tuple[Matrix6x6, ...] | list[Matrix6x6],
    S: Matrix6xn,
    num_samples: int = 200,
    polynomial_degree: int = POLYNOMIAL_DEGREE,
    freeze_stationary_joints: bool = False,
    joint_delta: Vectorn | None = None,
) -> EnergyEvaluation:
    dynamics_constants = precompute_inverse_dynamics_constants(
        M_LIST=M_LIST,
        G_LIST=G_LIST,
        S=S,
    )
    trajectory = sample_shaped_joint_trajectory(
        q_start=q_start,
        q_goal=q_goal,
        duration=duration,
        shape_params=shape_params,
        polynomial_degree=polynomial_degree,
        num_samples=num_samples,
        freeze_stationary_joints=freeze_stationary_joints,
        joint_delta=joint_delta,
    )
    tau_samples = []
    power_samples = []

    for q_k, q_dot_k, q_dot_dot_k in zip(trajectory.q, trajectory.q_dot, trajectory.q_dot_dot):
        Ftip_total = np.asarray(Ftip, dtype=float).reshape(6,)
        if payload_mass != 0.0:
            T_ee = fk(np.asarray(M_LIST[-1], dtype=float), S, q_k)
            R_ee = T_ee[:3, :3]
            f_space = -payload_mass * np.asarray(g, dtype=float).reshape(3,)
            f_ee = R_ee.T @ f_space
            Ftip_total = Ftip_total + np.hstack((np.zeros(3, dtype=float), f_ee))

        tau_k = inverse_dynamics(
            q=q_k,
            q_dot=q_dot_k,
            q_dot_dot=q_dot_dot_k,
            g=g,
            Ftip=Ftip_total,
            M_LIST=M_LIST,
            G_LIST=G_LIST,
            S=S,
            constants=dynamics_constants,
        )
        tau_samples.append(tau_k)
        power_samples.append(np.sum(np.maximum(tau_k * q_dot_k, 0.0)))

    tau = np.asarray(tau_samples, dtype=float)
    power = np.asarray(power_samples, dtype=float)
    energy = _integrate_power(power, trajectory.t)

    return EnergyEvaluation(
        energy=energy,
        t=trajectory.t,
        q=trajectory.q,
        q_dot=trajectory.q_dot,
        q_dot_dot=trajectory.q_dot_dot,
        tau=tau,
        power=power,
    )


def _direction_branch_candidates(
    q_start: Vectorn,
    q_goal: Vectorn,
    active_joint_indices: np.ndarray,
) -> list[tuple[Vectorn, tuple[int, ...]]]:
    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=float).reshape(-1)
    delta_options = _clockwise_counterclockwise_delta_options(q_start, q_goal)
    candidates: list[tuple[Vectorn, tuple[int, ...]]] = []

    for active_branch_bits in product((0, 1), repeat=active_joint_indices.size):
        branch_bits = [0] * q_start.size
        delta = delta_options[:, 0].copy()

        for joint_idx, bit in zip(active_joint_indices, active_branch_bits):
            branch_bits[int(joint_idx)] = int(bit)
            delta[int(joint_idx)] = delta_options[int(joint_idx), int(bit)]

        candidates.append((delta, tuple(branch_bits)))

    return candidates


def _optimize_direction_branch(args: tuple) -> BranchOptimizationResult:
    (
        q_start,
        q_goal,
        duration,
        g,
        Ftip,
        payload_mass,
        M_LIST,
        G_LIST,
        S,
        num_samples,
        method,
        polynomial_degree,
        freeze_stationary_joints,
        n_joints,
        n_shape_coefficients,
        active_joint_indices,
        shape_bounds,
        joint_delta,
        direction_branch_bits,
        joint3_limits,
    ) = args

    x0 = np.zeros(n_shape_coefficients * active_joint_indices.size, dtype=float)

    def expand_active_shape_params(active_shape_params: Vectorn) -> Vectorn:
        full_coeffs = np.zeros((n_joints, n_shape_coefficients), dtype=float)
        if active_joint_indices.size:
            full_coeffs[active_joint_indices, :] = np.asarray(
                active_shape_params,
                dtype=float,
            ).reshape(active_joint_indices.size, n_shape_coefficients)
        return full_coeffs.reshape(-1)

    branch_cost_history: list[float] = []

    def objective(x: Vectorn) -> float:
        full_shape_params = expand_active_shape_params(x)
        evaluation = evaluate_shaped_trajectory_energy(
            q_start=q_start,
            q_goal=q_goal,
            duration=duration,
            shape_params=full_shape_params,
            polynomial_degree=polynomial_degree,
            freeze_stationary_joints=freeze_stationary_joints,
            g=g,
            Ftip=Ftip,
            payload_mass=payload_mass,
            M_LIST=M_LIST,
            G_LIST=G_LIST,
            S=S,
            num_samples=num_samples,
            joint_delta=joint_delta,
        )
        cost = evaluation.energy
        branch_cost_history.append(cost)
        return cost

    constraints = ()
    if joint3_limits is not None:

        def joint3_limit_constraint(x: Vectorn) -> Vectorn:
            return _joint3_limit_constraint_values(
                q_start=q_start,
                q_goal=q_goal,
                duration=duration,
                shape_params=expand_active_shape_params(x),
                polynomial_degree=polynomial_degree,
                num_samples=num_samples,
                freeze_stationary_joints=freeze_stationary_joints,
                joint_delta=joint_delta,
                joint3_limits=joint3_limits,
            )

        constraints = ({"type": "ineq", "fun": joint3_limit_constraint},)

    minimize_kwargs = {
        "fun": objective,
        "x0": x0,
        "method": method,
        "bounds": shape_bounds,
    }
    if constraints:
        minimize_kwargs["constraints"] = constraints

    result = minimize(**minimize_kwargs)
    final_x = expand_active_shape_params(np.asarray(result.x, dtype=float))
    evaluation = evaluate_shaped_trajectory_energy(
        q_start=q_start,
        q_goal=q_goal,
        duration=duration,
        shape_params=final_x,
        polynomial_degree=polynomial_degree,
        freeze_stationary_joints=freeze_stationary_joints,
        g=g,
        Ftip=Ftip,
        payload_mass=payload_mass,
        M_LIST=M_LIST,
        G_LIST=G_LIST,
        S=S,
        num_samples=num_samples,
        joint_delta=joint_delta,
    )
    min_constraint_margin = np.inf
    if joint3_limits is not None:
        min_constraint_margin = float(
            np.min(
                _joint3_limit_constraint_values(
                    q_start=q_start,
                    q_goal=q_goal,
                    duration=duration,
                    shape_params=final_x,
                    polynomial_degree=polynomial_degree,
                    num_samples=num_samples,
                    freeze_stationary_joints=freeze_stationary_joints,
                    joint_delta=joint_delta,
                    joint3_limits=joint3_limits,
                )
            )
        )

    return BranchOptimizationResult(
        energy=evaluation.energy,
        shape_params=final_x,
        success=bool(result.success),
        message=str(result.message),
        evaluation=evaluation,
        joint_delta=joint_delta,
        direction_branch_bits=direction_branch_bits,
        iterations=int(getattr(result, "nit", 0)),
        evaluations=int(getattr(result, "nfev", 0)),
        cost_history=np.asarray(branch_cost_history, dtype=float),
        min_constraint_margin=min_constraint_margin,
    )


def optimize_trajectory_shape(
    q_start: Vectorn,
    q_goal: Vectorn,
    duration: float,
    g: Vector3,
    Ftip: Vector6,
    payload_mass: float,
    M_LIST: tuple[Matrix4x4, ...] | list[Matrix4x4],
    G_LIST: tuple[Matrix6x6, ...] | list[Matrix6x6],
    S: Matrix6xn,
    shape_bounds: tuple[float, float] = (-1.0, 1.0),
    num_samples: int = 200,
    method: str = "SLSQP",
    polynomial_degree: int = POLYNOMIAL_DEGREE,
    freeze_stationary_joints: bool = False,
    search_direction_branches: bool = False,
    parallel_direction_branches: bool = False,
    joint3_limits: tuple[float, float] | None = JOINT3_LIMITS,
) -> ShapeOptimizationResult:
    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=float).reshape(-1)
    n_joints = q_start.size
    if joint3_limits is not None:
        _validate_joint3_endpoint(q_start, "q_start", joint3_limits)
        _validate_joint3_endpoint(q_goal, "q_goal", joint3_limits)
        if method.lower() not in {"slsqp", "cobyla", "cobyqa", "trust-constr"}:
            raise ValueError("joint3_limits requires a constraint-capable optimizer method.")

    n_shape_coefficients = _num_shape_coefficients(polynomial_degree)
    active_joint_mask = np.ones(n_joints, dtype=bool)
    if freeze_stationary_joints:
        active_joint_mask = ~_stationary_joint_mask(q_start, q_goal)
    active_joint_indices = np.flatnonzero(active_joint_mask)
    x0 = np.zeros(n_shape_coefficients * active_joint_indices.size, dtype=float)
    nonstationary_joint_indices = np.flatnonzero(~_stationary_joint_mask(q_start, q_goal))
    branch_joint_indices = np.intersect1d(
        active_joint_indices,
        nonstationary_joint_indices,
        assume_unique=True,
    )
    default_joint_delta = _shortest_angular_delta(q_start, q_goal)
    default_branch_bits = tuple(0 for _ in range(n_joints))
    branch_candidates = [(default_joint_delta, default_branch_bits)]
    if search_direction_branches and branch_joint_indices.size:
        branch_candidates = _direction_branch_candidates(
            q_start=q_start,
            q_goal=q_goal,
            active_joint_indices=branch_joint_indices,
        )

    lower, upper = shape_bounds
    if lower >= upper:
        raise ValueError("shape_bounds must satisfy lower < upper.")

    bounds = [(lower, upper)] * x0.size
    all_cost_history: list[float] = []

    if x0.size == 0:
        joint_delta, direction_branch_bits = branch_candidates[0]
        best_evaluation = evaluate_shaped_trajectory_energy(
            q_start=q_start,
            q_goal=q_goal,
            duration=duration,
            shape_params=np.zeros(n_shape_coefficients * n_joints, dtype=float),
            polynomial_degree=polynomial_degree,
            freeze_stationary_joints=freeze_stationary_joints,
            g=g,
            Ftip=Ftip,
            payload_mass=payload_mass,
            M_LIST=M_LIST,
            G_LIST=G_LIST,
            S=S,
            num_samples=num_samples,
            joint_delta=joint_delta,
        )
        all_cost_history.append(best_evaluation.energy)
        success = True
        message = "No active shape parameters because all joints are stationary."
        if joint3_limits is not None:
            constraint_values = _joint3_limit_constraint_values(
                q_start=q_start,
                q_goal=q_goal,
                duration=duration,
                shape_params=np.zeros(n_shape_coefficients * n_joints, dtype=float),
                polynomial_degree=polynomial_degree,
                num_samples=num_samples,
                freeze_stationary_joints=freeze_stationary_joints,
                joint_delta=joint_delta,
                joint3_limits=joint3_limits,
            )
            if np.min(constraint_values) < -CONSTRAINT_TOL:
                success = False
                message = (
                    "No active shape parameters and the joint 3 limit constraint "
                    "is infeasible for this trajectory."
                )

        return ShapeOptimizationResult(
            shape_params=np.zeros(n_shape_coefficients * n_joints, dtype=float),
            energy=best_evaluation.energy,
            success=success,
            message=message,
            iterations=0,
            evaluations=1,
            cost_history=np.asarray(all_cost_history, dtype=float),
            best_cost_history=np.asarray(all_cost_history, dtype=float),
            evaluation=best_evaluation,
            joint_delta=joint_delta,
            direction_branch_bits=direction_branch_bits,
            standard_branch_energy=best_evaluation.energy,
            standard_branch_success=success,
            standard_branch_evaluation=best_evaluation,
        )

    best_result = None
    branch_args = [
        (
            q_start,
            q_goal,
            duration,
            g,
            Ftip,
            payload_mass,
            M_LIST,
            G_LIST,
            S,
            num_samples,
            method,
            polynomial_degree,
            freeze_stationary_joints,
            n_joints,
            n_shape_coefficients,
            active_joint_indices,
            bounds,
            joint_delta,
            direction_branch_bits,
            joint3_limits,
        )
        for joint_delta, direction_branch_bits in branch_candidates
    ]

    if parallel_direction_branches and len(branch_args) > 1:
        try:
            with ProcessPoolExecutor() as executor:
                branch_results = list(executor.map(_optimize_direction_branch, branch_args))
        except (OSError, BrokenProcessPool):
            branch_results = [_optimize_direction_branch(args) for args in branch_args]
    else:
        branch_results = [_optimize_direction_branch(args) for args in branch_args]

    standard_result = None
    total_iterations = 0
    total_evaluations = 0

    def branch_selection_key(branch_result: BranchOptimizationResult) -> tuple[bool, float, bool, float]:
        constraint_satisfied = branch_result.min_constraint_margin >= -CONSTRAINT_TOL
        constraint_violation = max(0.0, -branch_result.min_constraint_margin)
        return (
            constraint_satisfied,
            -constraint_violation,
            branch_result.success,
            -branch_result.energy,
        )

    for branch_result in branch_results:
        total_iterations += branch_result.iterations
        total_evaluations += branch_result.evaluations
        all_cost_history.extend(branch_result.cost_history.tolist())
        if branch_result.direction_branch_bits == default_branch_bits:
            standard_result = branch_result
        if (
            best_result is None
            or branch_selection_key(branch_result) > branch_selection_key(best_result)
        ):
            best_result = branch_result

    assert best_result is not None
    assert standard_result is not None
    message = best_result.message

    if search_direction_branches:
        message = (
            f"{message} Optimized {len(branch_candidates)} direction branches. "
            f"Selected direction branch bits: {best_result.direction_branch_bits}."
        )
    if joint3_limits is not None:
        min_observed_q3 = float(np.min(best_result.evaluation.q[:, 2]))
        max_observed_q3 = float(np.max(best_result.evaluation.q[:, 2]))
        message = (
            f"{message} Joint 3 range: "
            f"[{min_observed_q3:.6f}, {max_observed_q3:.6f}] rad."
        )

    return ShapeOptimizationResult(
        shape_params=best_result.shape_params,
        energy=best_result.energy,
        success=best_result.success,
        message=message,
        iterations=total_iterations,
        evaluations=total_evaluations,
        cost_history=np.asarray(all_cost_history, dtype=float),
        best_cost_history=np.minimum.accumulate(np.asarray(all_cost_history, dtype=float)),
        evaluation=best_result.evaluation,
        joint_delta=best_result.joint_delta,
        direction_branch_bits=best_result.direction_branch_bits,
        standard_branch_energy=standard_result.energy,
        standard_branch_success=standard_result.success,
        standard_branch_evaluation=standard_result.evaluation,
    )


if __name__ == "__main__":
    from robot.ur5e_parameters import (
        M_LIST,
        G_LIST,
        S,
        BASE_MASS,
        SHOULDER_MASS,
        ELBOW_MASS,
        FOREARM_MASS,
        WRIST_MASS,
        END_EFFECTOR_MASS,
        BASE_COM,
        SHOULDER_COM,
        ELBOW_COM,
        FOREARM_COM,
        WRIST_COM,
        END_EFFECTOR_COM,
    )

    cubic_output_path = Path("trajectories/cubic_trajectory.json")
    optimized_output_path = Path("trajectories/optimized_trajectory.json")
    cubic_output_path.parent.mkdir(parents=True, exist_ok=True)

    q_start = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0], dtype=float)
    q_goal = np.array([np.pi, 0.0, -np.pi/2, np.pi, 0.0, 0.0], dtype=float)
    # q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    # q_goal = np.array([0.0, np.pi+0.0, np.pi/3, 0.0, 0.0, 0.0], dtype=float)

    _validate_joint3_endpoint(q_start, "q_start")
    _validate_joint3_endpoint(q_goal, "q_goal")

    duration = 5.0
    polynomial_degree = 6
    g = np.array([0.0, 0.0, -9.81])
    Ftip = np.zeros(6)
    payload_mass = 10

    pe_comparison = compare_potential_energy(
        q_start=q_start,
        q_goal=q_goal,
        g=g,
        M_LIST=M_LIST,
        S=S,
        link_masses=np.array(
            [
                BASE_MASS,
                SHOULDER_MASS,
                ELBOW_MASS,
                FOREARM_MASS,
                WRIST_MASS,
                END_EFFECTOR_MASS,
            ],
            dtype=float,
        ),
        link_com_positions=np.array(
            [
                BASE_COM,
                SHOULDER_COM,
                ELBOW_COM,
                FOREARM_COM,
                WRIST_COM,
                END_EFFECTOR_COM,
            ],
            dtype=float,
        ),
        link_names=("base", "shoulder", "elbow", "forearm", "wrist", "end_effector"),
        payload_mass=payload_mass,
    )
    print(f"Start PE: {pe_comparison.start.total_potential_energy:.6f} J")
    print(f"Goal PE: {pe_comparison.goal.total_potential_energy:.6f} J")
    print(f"Delta PE: {pe_comparison.delta_potential_energy:.6f} J")

    cubic_evaluation = evaluate_shaped_trajectory_energy(
        q_start=q_start,
        q_goal=q_goal,
        duration=duration,
        shape_params=np.zeros(_num_shape_coefficients(polynomial_degree) * q_start.size),
        polynomial_degree=polynomial_degree,
        g=g,
        Ftip=Ftip,
        payload_mass=payload_mass,
        M_LIST=M_LIST,
        G_LIST=G_LIST,
        S=S,
        num_samples=100,
    )

    optimization_result = optimize_trajectory_shape(
        q_start=q_start,
        q_goal=q_goal,
        duration=duration,
        polynomial_degree=polynomial_degree,
        freeze_stationary_joints=True,
        search_direction_branches=True,
        parallel_direction_branches=True,
        g=g,
        Ftip=Ftip,
        payload_mass=payload_mass,
        M_LIST=M_LIST,
        G_LIST=G_LIST,
        S=S,
        shape_bounds=(-1000.0, 1000.0),
        num_samples=100,
        method="SLSQP",
    )

    print("Optimization Result:")
    print("Method: SLSQP")
    print(f"Success: {optimization_result.success}")
    print(f"Polynomial Degree: {polynomial_degree}")
    print(f"Baseline 5th Degree Energy: {cubic_evaluation.energy:.4f}")
    print(f"Optimized Energy: {optimization_result.energy:.4f}")
    print(f"Selected Direction Branch Bits: {optimization_result.direction_branch_bits}")
    if optimization_result.standard_branch_evaluation is not None:
        print("Standard Branch Result:")
        print(f"  Success: {optimization_result.standard_branch_success}")
        print(f"  Energy: {optimization_result.standard_branch_energy:.4f}")

    print(f"Recorded Objective Evaluations: {optimization_result.cost_history.size}")

    save_joint_trajectory(cubic_output_path, cubic_evaluation)
    save_joint_trajectory(optimized_output_path, optimization_result.evaluation)

    if optimization_result.success:
        plot_joint_trajectory_comparison(
            cubic_evaluation,
            optimization_result.evaluation,
            polynomial_degree=polynomial_degree,
        )
    plot_power_comparison(
        cubic_evaluation,
        optimization_result.evaluation,
        polynomial_degree=polynomial_degree,
        standard_branch_evaluation=optimization_result.standard_branch_evaluation,
    )
    plot_cost_history(optimization_result)
