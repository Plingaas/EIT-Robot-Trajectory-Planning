from dataclasses import dataclass
from itertools import product

import numpy as np

from core.dynamics import inverse_dynamics, precompute_inverse_dynamics_constants
from core.kinematics.fk import fk
from core.trajectory_optimization import EnergyEvaluation
from core.types import Matrix4x4, Matrix6x6, Matrix6xn, Vector3, Vector6, Vectorn


@dataclass(frozen=True)
class BaselineWindingBranchResult:
    branch_bits: tuple[int, ...]
    direction_labels: tuple[str, ...]
    delta: Vectorn
    effective_goal: Vectorn
    energy: float
    peak_power: float
    max_abs_tau: float
    evaluation: EnergyEvaluation


def shortest_angular_delta(q_start: Vectorn, q_goal: Vectorn) -> Vectorn:
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    return (q_goal - q_start + np.pi) % (2.0 * np.pi) - np.pi


def clockwise_counterclockwise_deltas(q_start: Vectorn, q_goal: Vectorn) -> np.ndarray:
    short = shortest_angular_delta(q_start, q_goal)
    opposite = np.where(short >= 0.0, short - 2.0 * np.pi, short + 2.0 * np.pi)
    return np.column_stack((short, opposite))


def sample_baseline_trajectory_from_delta(
    q_start: Vectorn,
    delta: Vectorn,
    duration: float,
    num_samples: int,
) -> tuple[Vectorn, np.ndarray, np.ndarray, np.ndarray]:
    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    delta = np.asarray(delta, dtype=float).reshape(-1)

    if q_start.shape != delta.shape:
        raise ValueError("q_start and delta must have the same shape.")
    if duration <= 0.0:
        raise ValueError("duration must be > 0.")
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")

    t = np.linspace(0.0, duration, num_samples, dtype=float)
    u = t / duration

    h = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    h_u = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    h_uu = 60.0 * u - 180.0 * u**2 + 120.0 * u**3

    q = q_start[None, :] + h[:, None] * delta[None, :]
    q_dot = h_u[:, None] * delta[None, :] / duration
    q_dot_dot = h_uu[:, None] * delta[None, :] / (duration**2)

    return t, q, q_dot, q_dot_dot


def joint_power(tau: Vectorn, q_dot: Vectorn, power_mode: str) -> float:
    joint_power_samples = np.asarray(tau, dtype=float) * np.asarray(q_dot, dtype=float)

    if power_mode == "positive":
        return float(np.sum(np.maximum(joint_power_samples, 0.0)))
    if power_mode == "signed":
        return float(np.sum(joint_power_samples))
    if power_mode == "absolute":
        return float(np.sum(np.abs(joint_power_samples)))

    raise ValueError("power_mode must be one of: 'positive', 'signed', 'absolute'.")


def evaluate_baseline_branch_energy(
    q_start: Vectorn,
    delta: Vectorn,
    duration: float,
    g: Vector3,
    Ftip: Vector6,
    payload_mass: float,
    M_LIST: tuple[Matrix4x4, ...] | list[Matrix4x4],
    G_LIST: tuple[Matrix6x6, ...] | list[Matrix6x6],
    S: Matrix6xn,
    num_samples: int = 100,
    power_mode: str = "positive",
) -> EnergyEvaluation:
    t, q, q_dot, q_dot_dot = sample_baseline_trajectory_from_delta(
        q_start=q_start,
        delta=delta,
        duration=duration,
        num_samples=num_samples,
    )
    dynamics_constants = precompute_inverse_dynamics_constants(
        M_LIST=M_LIST,
        G_LIST=G_LIST,
        S=S,
    )

    tau_samples = []
    power_samples = []

    for q_k, q_dot_k, q_dot_dot_k in zip(q, q_dot, q_dot_dot):
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
        power_samples.append(joint_power(tau_k, q_dot_k, power_mode))

    tau = np.asarray(tau_samples, dtype=float)
    power = np.asarray(power_samples, dtype=float)
    energy = float(np.trapezoid(power, t))

    return EnergyEvaluation(
        energy=energy,
        t=t,
        q=q,
        q_dot=q_dot,
        q_dot_dot=q_dot_dot,
        tau=tau,
        power=power,
    )


def evaluate_all_baseline_winding_branches(
    q_start: Vectorn,
    q_goal: Vectorn,
    duration: float,
    g: Vector3,
    Ftip: Vector6,
    payload_mass: float,
    M_LIST: tuple[Matrix4x4, ...] | list[Matrix4x4],
    G_LIST: tuple[Matrix6x6, ...] | list[Matrix6x6],
    S: Matrix6xn,
    num_samples: int = 100,
    power_mode: str = "positive",
    branch_joint_indices: tuple[int, ...] | list[int] | None = None,
) -> list[BaselineWindingBranchResult]:
    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=float).reshape(-1)

    if q_start.shape != q_goal.shape:
        raise ValueError("q_start and q_goal must have the same shape.")

    if branch_joint_indices is None:
        branch_joint_indices = tuple(range(q_start.size))
    else:
        branch_joint_indices = tuple(int(index) for index in branch_joint_indices)

    invalid_indices = [index for index in branch_joint_indices if index < 0 or index >= q_start.size]
    if invalid_indices:
        raise ValueError(f"Invalid branch joint indices: {invalid_indices}")

    delta_options = clockwise_counterclockwise_deltas(q_start, q_goal)
    shortest_delta = delta_options[:, 0]
    results: list[BaselineWindingBranchResult] = []

    for active_branch_bits in product((0, 1), repeat=len(branch_joint_indices)):
        active_branch_bits = tuple(int(bit) for bit in active_branch_bits)
        branch_bits_list = [0] * q_start.size
        delta = shortest_delta.copy()
        for joint_idx, bit in zip(branch_joint_indices, active_branch_bits):
            branch_bits_list[joint_idx] = bit
            delta[joint_idx] = delta_options[joint_idx, bit]

        branch_bits = tuple(branch_bits_list)
        direction_labels = tuple("shortest" if bit == 0 else "opposite" for bit in branch_bits)
        evaluation = evaluate_baseline_branch_energy(
            q_start=q_start,
            delta=delta,
            duration=duration,
            g=g,
            Ftip=Ftip,
            payload_mass=payload_mass,
            M_LIST=M_LIST,
            G_LIST=G_LIST,
            S=S,
            num_samples=num_samples,
            power_mode=power_mode,
        )
        results.append(
            BaselineWindingBranchResult(
                branch_bits=branch_bits,
                direction_labels=direction_labels,
                delta=delta,
                effective_goal=q_start + delta,
                energy=evaluation.energy,
                peak_power=float(np.max(evaluation.power)),
                max_abs_tau=float(np.max(np.abs(evaluation.tau))),
                evaluation=evaluation,
            )
        )

    return sorted(results, key=lambda result: result.energy)


def print_baseline_winding_branch_results(
    results: list[BaselineWindingBranchResult],
    limit: int | None = None,
) -> None:
    shown_results = results if limit is None else results[:limit]
    print(f"{'rank':>4}  {'energy [J]':>12}  {'peak P [W]':>12}  {'max |tau|':>12}  branch bits  delta")
    for rank, result in enumerate(shown_results, start=1):
        delta_text = np.array2string(result.delta, precision=3, suppress_small=True)
        print(
            f"{rank:4d}  "
            f"{result.energy:12.6f}  "
            f"{result.peak_power:12.6f}  "
            f"{result.max_abs_tau:12.6f}  "
            f"{result.branch_bits}  "
            f"{delta_text}"
        )


if __name__ == "__main__":
    from robot.ur5e_parameters import G_LIST, M_LIST, S

    q_start = np.array([0.0, np.pi + 0.01, 0.0, 0.0, 0.0, 0.0], dtype=float)
    q_goal = np.zeros(6, dtype=float)

    branch_results = evaluate_all_baseline_winding_branches(
        q_start=q_start,
        q_goal=q_goal,
        duration=5.0,
        g=np.array([0.0, 0.0, -9.81], dtype=float),
        Ftip=np.zeros(6, dtype=float),
        payload_mass=10.0,
        M_LIST=M_LIST,
        G_LIST=G_LIST,
        S=S,
        num_samples=100,
        power_mode="positive",
        branch_joint_indices=(0, 1, 2),
    )
    print_baseline_winding_branch_results(branch_results)
