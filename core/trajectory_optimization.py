import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from core.dynamics import inverse_dynamics
from core.kinematics.fk import fk
from core.types import Matrix4x4, Matrix6x6, Matrix6xn, Vector3, Vector6, Vectorn

POLYNOMIAL_DEGREE = 7


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
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    n_joints = cubic_evaluation.q.shape[1]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    axes = axes.ravel()

    for joint_idx in range(n_joints):
        ax = axes[joint_idx]
        ax.plot(cubic_evaluation.t, cubic_evaluation.q[:, joint_idx], label="Cubic", linewidth=2.0)
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
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

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
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    cubic_cumulative = _cumulative_integral(cubic_evaluation.power, cubic_evaluation.t)
    optimized_cumulative = _cumulative_integral(optimized_evaluation.power, optimized_evaluation.t)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(cubic_evaluation.t, cubic_evaluation.power, label="Cubic", linewidth=2.0)
    axes[0].plot(
        optimized_evaluation.t,
        optimized_evaluation.power,
        label=f"Optimized {_ordinal(polynomial_degree)} Degree",
        linewidth=2.0,
    )
    axes[0].set_title("Power Consumption")
    axes[0].set_ylabel("Power")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(cubic_evaluation.t, cubic_cumulative, label="Cubic", linewidth=2.0)
    axes[1].plot(
        optimized_evaluation.t,
        optimized_cumulative,
        label=f"Optimized {_ordinal(polynomial_degree)} Degree",
        linewidth=2.0,
    )
    axes[1].set_title("Cumulative Energy")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Energy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def _integrate_samples(power: Vectorn, time: Vectorn) -> float:
    return float(np.trapezoid(power, time))


def _num_shape_coefficients(polynomial_degree: int) -> int:
    if polynomial_degree < 3:
        raise ValueError("polynomial_degree must be at least 3.")
    return polynomial_degree - 3


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

    for p in range(2, polynomial_degree - 1):
        basis_values.append(u**p - 2.0 * u**(p + 1) + u**(p + 2))
        basis_first.append(
            p * u**(p - 1)
            - 2.0 * (p + 1) * u**p
            + (p + 2) * u**(p + 1)
        )
        basis_second.append(
            p * (p - 1) * u**(p - 2)
            - 2.0 * p * (p + 1) * u**(p - 1)
            + (p + 1) * (p + 2) * u**p
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

    t = np.linspace(0.0, duration, num_samples, dtype=float)
    u = t / duration

    # Base cubic: fixed endpoints with zero endpoint velocities.
    h = 3.0 * u**2 - 2.0 * u**3
    h_u = 6.0 * u - 6.0 * u**2
    h_uu = 6.0 - 12.0 * u

    # Boundary-safe shape terms: phi(0)=phi(1)=phi'(0)=phi'(1)=0.
    phi, phi_u, phi_uu = _shape_basis(u, polynomial_degree)

    dq = q_goal - q_start
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
) -> EnergyEvaluation:
    trajectory = sample_shaped_joint_trajectory(
        q_start=q_start,
        q_goal=q_goal,
        duration=duration,
        shape_params=shape_params,
        polynomial_degree=polynomial_degree,
        num_samples=num_samples,
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
        )
        tau_samples.append(tau_k)
        power_samples.append(float(np.sum(np.abs(tau_k * q_dot_k))))

    tau = np.asarray(tau_samples, dtype=float)
    power = np.asarray(power_samples, dtype=float)
    energy = _integrate_samples(power, trajectory.t)

    return EnergyEvaluation(
        energy=energy,
        t=trajectory.t,
        q=trajectory.q,
        q_dot=trajectory.q_dot,
        q_dot_dot=trajectory.q_dot_dot,
        tau=tau,
        power=power,
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
    method: str = "Powell",
    maxiter: int = 20,
    maxfev: int = 200,
    polynomial_degree: int = POLYNOMIAL_DEGREE,
) -> ShapeOptimizationResult:
    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=float).reshape(-1)
    n_joints = q_start.size
    x0 = np.zeros(_num_shape_coefficients(polynomial_degree) * n_joints, dtype=float)

    lower, upper = shape_bounds
    if lower >= upper:
        raise ValueError("shape_bounds must satisfy lower < upper.")

    bounds = [(lower, upper)] * x0.size
    cost_history: list[float] = []

    def objective(x: Vectorn) -> float:
        evaluation = evaluate_shaped_trajectory_energy(
            q_start=q_start,
            q_goal=q_goal,
            duration=duration,
            shape_params=x,
            polynomial_degree=polynomial_degree,
            g=g,
            Ftip=Ftip,
            payload_mass=payload_mass,
            M_LIST=M_LIST,
            G_LIST=G_LIST,
            S=S,
            num_samples=num_samples,
        )
        cost = evaluation.energy
        cost_history.append(cost)
        return cost

    result = minimize(
        objective,
        x0=x0,
        method=method,
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "maxfev": maxfev,
        },
    )

    best_evaluation = evaluate_shaped_trajectory_energy(
        q_start=q_start,
        q_goal=q_goal,
        duration=duration,
        shape_params=result.x,
        polynomial_degree=polynomial_degree,
        g=g,
        Ftip=Ftip,
        payload_mass=payload_mass,
        M_LIST=M_LIST,
        G_LIST=G_LIST,
        S=S,
        num_samples=num_samples,
    )

    return ShapeOptimizationResult(
        shape_params=np.asarray(result.x, dtype=float),
        energy=best_evaluation.energy,
        success=bool(result.success),
        message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        evaluations=int(getattr(result, "nfev", 0)),
        cost_history=np.asarray(cost_history, dtype=float),
        best_cost_history=np.minimum.accumulate(np.asarray(cost_history, dtype=float)),
        evaluation=best_evaluation,
    )


if __name__ == "__main__":
    from robot.ur5e_parameters import M_LIST, G_LIST, S

    cubic_output_path = Path("trajectories/cubic_trajectory.json")
    optimized_output_path = Path("trajectories/optimized_trajectory.json")
    cubic_output_path.parent.mkdir(parents=True, exist_ok=True)

    q_start = np.array([0.0, -np.pi, 0.0, 0.0, 0.0, 0.0], dtype=float)
    q_goal = np.zeros(6, dtype=float)
    duration = 5.0
    polynomial_degree = 7
    g = np.array([0.0, 0.0, -9.81])
    Ftip = np.zeros(6)
    payload_mass = 5

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
        g=g,
        Ftip=Ftip,
        payload_mass=payload_mass,
        M_LIST=M_LIST,
        G_LIST=G_LIST,
        S=S,
        shape_bounds=(-1000.0, 1000.0),
        num_samples=100,
        method="SLSQP",
        maxiter=1000,
        maxfev=5000,
    )

    print("Optimization Result:")
    print(f"Success: {optimization_result.success}")
    print(f"Message: {optimization_result.message}")
    print(f"Polynomial Degree: {polynomial_degree}")
    print(f"Cubic Energy: {cubic_evaluation.energy:.4f}")
    print(f"Optimized Energy: {optimization_result.energy:.4f}")
    print(f"Recorded Objective Evaluations: {optimization_result.cost_history.size}")

    save_joint_trajectory(cubic_output_path, cubic_evaluation)
    save_joint_trajectory(optimized_output_path, optimization_result.evaluation)
    print(f"Saved cubic trajectory to: {cubic_output_path}")
    print(f"Saved optimized trajectory to: {optimized_output_path}")

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
    )
    plot_cost_history(optimization_result)

# Cubic Energy: 76.7894
# Optimized Energy: 76.6235
# 76.4465
# 76.4254

# Cubic Energy: 76.7122
# Optimized Energy: 76.1659
