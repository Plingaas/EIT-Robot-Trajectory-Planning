from dataclasses import dataclass

import numpy as np

from core.kinematics.fk import fk_links
from core.types import Matrix4x4, Matrix6xn, Vector3, Vectorn


@dataclass(frozen=True)
class LinkPotentialEnergy:
    link_name: str
    mass: float
    com_world: Vector3
    potential_energy: float


@dataclass(frozen=True)
class PotentialEnergyBreakdown:
    total_potential_energy: float
    links: tuple[LinkPotentialEnergy, ...]


@dataclass(frozen=True)
class PotentialEnergyComparison:
    start: PotentialEnergyBreakdown
    goal: PotentialEnergyBreakdown
    delta_potential_energy: float
    delta_per_link: tuple[float, ...]


def compute_potential_energy_breakdown(
    q: Vectorn,
    g: Vector3,
    M_LIST: tuple[Matrix4x4, ...] | list[Matrix4x4],
    S: Matrix6xn,
    link_masses: Vectorn,
    link_com_positions: np.ndarray,
    link_names: tuple[str, ...] | list[str] | None = None,
    payload_mass: float = 0.0,
) -> PotentialEnergyBreakdown:
    q = np.asarray(q, dtype=float).reshape(-1)
    g = np.asarray(g, dtype=float).reshape(3,)
    link_masses = np.asarray(link_masses, dtype=float).reshape(-1)
    link_com_positions = np.asarray(link_com_positions, dtype=float)

    if len(M_LIST) != q.size + 1:
        raise ValueError("M_LIST must contain q.size + 1 link home frames.")
    if link_masses.size != q.size:
        raise ValueError("link_masses must contain one mass per moving link.")
    if link_com_positions.shape != (q.size, 3):
        raise ValueError("link_com_positions must have shape (q.size, 3).")
    if payload_mass < 0.0:
        raise ValueError("payload_mass must be >= 0.")

    if link_names is None:
        link_names = tuple(f"link_{idx + 1}" for idx in range(q.size))
    elif len(link_names) != q.size:
        raise ValueError("link_names must contain one name per moving link.")

    effective_link_masses = link_masses.copy()
    effective_link_masses[-1] += float(payload_mass)

    gravity_magnitude = float(np.linalg.norm(g))
    gravity_direction = np.zeros(3, dtype=float)
    if gravity_magnitude > 0.0:
        gravity_direction = g / gravity_magnitude

    link_transforms = fk_links(list(M_LIST), S, q)
    links: list[LinkPotentialEnergy] = []

    for name, mass, link_transform, com_local in zip(
        link_names,
        effective_link_masses,
        link_transforms[1:],
        link_com_positions,
    ):
        com_world_h = link_transform @ np.hstack((com_local, 1.0))
        com_world = com_world_h[:3]
        height_along_up = -float(gravity_direction @ com_world)
        potential_energy = float(mass * gravity_magnitude * height_along_up)
        links.append(
            LinkPotentialEnergy(
                link_name=str(name),
                mass=float(mass),
                com_world=com_world,
                potential_energy=potential_energy,
            )
        )
    total_potential_energy = float(sum(link.potential_energy for link in links))
    return PotentialEnergyBreakdown(
        total_potential_energy=total_potential_energy,
        links=tuple(links),
    )


def compare_potential_energy(
    q_start: Vectorn,
    q_goal: Vectorn,
    g: Vector3,
    M_LIST: tuple[Matrix4x4, ...] | list[Matrix4x4],
    S: Matrix6xn,
    link_masses: Vectorn,
    link_com_positions: np.ndarray,
    link_names: tuple[str, ...] | list[str] | None = None,
    payload_mass: float = 0.0,
) -> PotentialEnergyComparison:
    start = compute_potential_energy_breakdown(
        q=q_start,
        g=g,
        M_LIST=M_LIST,
        S=S,
        link_masses=link_masses,
        link_com_positions=link_com_positions,
        link_names=link_names,
        payload_mass=payload_mass,
    )
    goal = compute_potential_energy_breakdown(
        q=q_goal,
        g=g,
        M_LIST=M_LIST,
        S=S,
        link_masses=link_masses,
        link_com_positions=link_com_positions,
        link_names=link_names,
        payload_mass=payload_mass,
    )
    delta_per_link = tuple(
        goal_link.potential_energy - start_link.potential_energy
        for start_link, goal_link in zip(start.links, goal.links)
    )
    delta_potential_energy = float(
        goal.total_potential_energy - start.total_potential_energy
    )
    return PotentialEnergyComparison(
        start=start,
        goal=goal,
        delta_potential_energy=delta_potential_energy,
        delta_per_link=delta_per_link,
    )
