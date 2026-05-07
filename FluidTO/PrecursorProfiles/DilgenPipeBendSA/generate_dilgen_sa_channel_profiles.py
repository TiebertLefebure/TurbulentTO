#!/usr/bin/env python3
"""Generate fully developed SA channel profiles for the Dilgen bend inlet.

Dilgen et al. (2018), Sec. 5.1, use a 2D 90-degree channel bend with
half inlet height H = 0.1 m, bulk velocity Ub = 5 m/s, and kinematic
viscosity nu = 5e-5 m^2/s. The paper states that the inlet velocity and
turbulence profiles are obtained from an a-priori fully developed
turbulent simulation, but it does not tabulate those profiles.

This script provides the matching precursor step for this repository's
Spalart-Allmaras implementation. It solves the 1D fully developed plane
channel equations with the same no-ft2 SA closure used in
TurbulenceModel_SpalartAllmaras_TO_Frozen.py, adjusting the streamwise
pressure gradient every nonlinear iteration so that the bulk velocity is
exactly the requested value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz


CB1 = 0.1355
CB2 = 0.622
SIGMA = 2.0 / 3.0
KAPPA = 0.41
CW2 = 0.3
CW3 = 2.0
CV1 = 7.1
CW1 = CB1 / KAPPA**2 + (1.0 + CB2) / SIGMA


def stretched_channel_nodes(height: float, point_count: int) -> np.ndarray:
    """Cosine-clustered nodes, including both walls."""
    if point_count < 33:
        raise ValueError("Use at least 33 points for the channel profile.")
    eta = np.linspace(0.0, 1.0, int(point_count))
    return 0.5 * height * (1.0 - np.cos(np.pi * eta))


def face_average(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 0.5 * (left + right)


def solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Thomas solve for a tridiagonal linear system."""
    n = diag.size
    c_prime = np.zeros(max(n - 1, 0), dtype=float)
    d_prime = np.zeros(n, dtype=float)

    pivot = diag[0]
    if abs(pivot) < 1.0e-300:
        raise FloatingPointError("Zero pivot in tridiagonal solve.")
    if n > 1:
        c_prime[0] = upper[0] / pivot
    d_prime[0] = rhs[0] / pivot

    for i in range(1, n):
        pivot = diag[i] - lower[i - 1] * c_prime[i - 1]
        if abs(pivot) < 1.0e-300:
            raise FloatingPointError("Zero pivot in tridiagonal solve.")
        if i < n - 1:
            c_prime[i] = upper[i] / pivot
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / pivot

    solution = np.zeros(n, dtype=float)
    solution[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    return solution


def solve_dirichlet_reaction_diffusion(
    y: np.ndarray,
    diffusivity: np.ndarray,
    source: np.ndarray,
    reaction: np.ndarray | None = None,
) -> np.ndarray:
    """Solve -d/dy(a dphi/dy) + r phi = s with phi=0 at both walls."""
    n = y.size
    interior = n - 2
    if interior <= 0:
        raise ValueError("At least three nodes are required.")

    dy_left = y[1:-1] - y[:-2]
    dy_right = y[2:] - y[1:-1]
    volumes = 0.5 * (dy_left + dy_right)
    a_left = face_average(diffusivity[1:-1], diffusivity[:-2])
    a_right = face_average(diffusivity[1:-1], diffusivity[2:])

    lower = -a_left[1:] / dy_left[1:]
    upper = -a_right[:-1] / dy_right[:-1]
    diag = a_left / dy_left + a_right / dy_right
    if reaction is not None:
        diag = diag + reaction[1:-1] * volumes
    rhs = source[1:-1] * volumes

    interior_solution = solve_tridiagonal(lower, diag, upper, rhs)
    solution = np.zeros(n, dtype=float)
    solution[1:-1] = interior_solution
    return solution


def sa_eddy_viscosity(nu_tilde: np.ndarray, nu: float) -> np.ndarray:
    chi = np.maximum(nu_tilde / nu, 0.0)
    fv1 = chi**3 / (chi**3 + CV1**3 + 1.0e-300)
    return np.maximum(nu_tilde, 0.0) * fv1


def solve_velocity_for_bulk(
    y: np.ndarray,
    nu_effective: np.ndarray,
    bulk_velocity: float,
) -> tuple[np.ndarray, float]:
    """Return u(y) and the pressure-gradient source G = -dpdx/rho."""
    unit_source = np.ones_like(y)
    unit_velocity = solve_dirichlet_reaction_diffusion(y, nu_effective, unit_source)
    unit_bulk = _trapezoid(unit_velocity, y) / (y[-1] - y[0])
    if unit_bulk <= 0.0:
        raise FloatingPointError("Unit-pressure channel solve produced non-positive bulk flow.")
    pressure_source = float(bulk_velocity) / unit_bulk
    return unit_velocity * pressure_source, pressure_source


def solve_sa_update(
    y: np.ndarray,
    velocity: np.ndarray,
    nu_tilde_old: np.ndarray,
    nu: float,
    nu_tilde_floor: float,
) -> np.ndarray:
    height = y[-1] - y[0]
    wall_distance = np.maximum(np.minimum(y - y[0], y[-1] - y), 1.0e-12 * height)
    dudy = np.gradient(velocity, y, edge_order=2)
    dnu_dy = np.gradient(nu_tilde_old, y, edge_order=2)

    nu_tilde_safe = np.maximum(nu_tilde_old, nu_tilde_floor)
    chi = nu_tilde_safe / nu
    fv1 = chi**3 / (chi**3 + CV1**3 + 1.0e-300)
    fv2 = 1.0 - chi / (1.0 + chi * fv1 + 1.0e-300)

    strain = np.abs(dudy)
    s_tilde = strain + nu_tilde_safe / (KAPPA**2 * wall_distance**2) * fv2
    s_tilde = np.maximum(s_tilde, 1.0e-12)

    r_arg = nu_tilde_safe / (s_tilde * KAPPA**2 * wall_distance**2 + 1.0e-300)
    r_value = np.minimum(r_arg, 10.0)
    g_value = r_value + CW2 * (r_value**6 - r_value)
    fw = g_value * ((1.0 + CW3**6) / (g_value**6 + CW3**6 + 1.0e-300)) ** (1.0 / 6.0)

    production = CB1 * s_tilde * nu_tilde_safe
    cross_diffusion = (CB2 / SIGMA) * dnu_dy**2
    reaction = CW1 * fw * nu_tilde_safe / wall_distance**2
    diffusivity = (nu + nu_tilde_safe) / SIGMA

    source = production + cross_diffusion
    updated = solve_dirichlet_reaction_diffusion(y, diffusivity, source, reaction)
    updated[0] = 0.0
    updated[-1] = 0.0
    return np.maximum(updated, 0.0)


def generate_profile(
    bulk_velocity: float,
    half_height: float,
    nu: float,
    point_count: int,
    max_iterations: int,
    relaxation: float,
    tolerance: float,
    initial_mut_ratio: float,
    max_nu_tilde_ratio: float | None,
) -> dict[str, np.ndarray | float | int]:
    height = 2.0 * half_height
    y = stretched_channel_nodes(height, point_count)
    shape = np.sin(np.pi * y / height) ** 2
    nu_tilde = np.maximum(initial_mut_ratio * nu * shape, 0.0)
    nu_tilde[0] = 0.0
    nu_tilde[-1] = 0.0
    nu_tilde_floor = max(1.0e-14 * nu, 1.0e-18)
    nu_tilde_ceiling = None if max_nu_tilde_ratio is None else max_nu_tilde_ratio * nu

    last_residual = math.inf
    velocity = np.zeros_like(y)
    pressure_source = math.nan

    for iteration in range(1, max_iterations + 1):
        nu_t = sa_eddy_viscosity(nu_tilde, nu)
        velocity, pressure_source = solve_velocity_for_bulk(y, nu + nu_t, bulk_velocity)
        nu_tilde_new = solve_sa_update(y, velocity, nu_tilde, nu, nu_tilde_floor)
        if nu_tilde_ceiling is not None:
            nu_tilde_new = np.minimum(nu_tilde_new, nu_tilde_ceiling)

        relaxed = (1.0 - relaxation) * nu_tilde + relaxation * nu_tilde_new
        relaxed[0] = 0.0
        relaxed[-1] = 0.0
        scale = max(float(np.max(np.abs(relaxed))), nu)
        last_residual = float(np.max(np.abs(relaxed - nu_tilde)) / scale)
        nu_tilde = relaxed
        if last_residual < tolerance:
            break

    nu_t = sa_eddy_viscosity(nu_tilde, nu)
    velocity, pressure_source = solve_velocity_for_bulk(y, nu + nu_t, bulk_velocity)
    dudy = np.gradient(velocity, y, edge_order=2)
    bulk = _trapezoid(velocity, y) / height
    return {
        "y": y,
        "u_x": velocity,
        "u_y": np.zeros_like(y),
        "nu_tilde": nu_tilde,
        "nu_t": nu_t,
        "nu_t_over_nu": nu_t / nu,
        "wall_distance": np.minimum(y, height - y),
        "du_dy": dudy,
        "bulk_velocity": float(bulk),
        "pressure_gradient_over_density": float(pressure_source),
        "iterations": int(iteration),
        "residual": float(last_residual),
    }


def write_profile_csv(path: Path, profile: dict[str, np.ndarray | float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["y", "u_x", "u_y", "nu_tilde", "nu_t", "nu_t_over_nu", "wall_distance", "du_dy"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        row_count = len(profile["y"])  # type: ignore[arg-type]
        for idx in range(row_count):
            writer.writerow(["{:.16e}".format(float(profile[name][idx])) for name in columns])  # type: ignore[index]


def write_metadata(path: Path, args: argparse.Namespace, profile: dict[str, np.ndarray | float | int]) -> None:
    metadata = {
        "bulk_velocity_target": args.bulk_velocity,
        "bulk_velocity_integrated": profile["bulk_velocity"],
        "half_height": args.half_height,
        "height": 2.0 * args.half_height,
        "kinematic_viscosity": args.kinematic_viscosity,
        "reynolds_number_half_height": args.bulk_velocity * args.half_height / args.kinematic_viscosity,
        "point_count": args.points,
        "iterations": profile["iterations"],
        "residual": profile["residual"],
        "pressure_gradient_over_density": profile["pressure_gradient_over_density"],
        "model": "1D fully developed plane-channel Spalart-Allmaras no-ft2 precursor",
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).resolve().parent / "dilgen_sa_channel_profile.csv"
    parser.add_argument("--bulk-velocity", type=float, default=5.0)
    parser.add_argument("--half-height", type=float, default=0.1)
    parser.add_argument("--kinematic-viscosity", type=float, default=5.0e-5)
    parser.add_argument("--points", type=int, default=513)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--relaxation", type=float, default=0.2)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--initial-mut-ratio", type=float, default=30.0)
    parser.add_argument(
        "--max-nu-tilde-ratio",
        type=float,
        default=100.0,
        help="Clip nu_tilde to this multiple of nu; use a negative value to disable.",
    )
    parser.add_argument("--output", type=Path, default=default_output)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bulk_velocity <= 0.0:
        raise ValueError("--bulk-velocity must be positive.")
    if args.half_height <= 0.0:
        raise ValueError("--half-height must be positive.")
    if args.kinematic_viscosity <= 0.0:
        raise ValueError("--kinematic-viscosity must be positive.")
    if not 0.0 < args.relaxation <= 1.0:
        raise ValueError("--relaxation must be in (0, 1].")
    max_ratio = None if args.max_nu_tilde_ratio < 0.0 else args.max_nu_tilde_ratio

    profile = generate_profile(
        bulk_velocity=args.bulk_velocity,
        half_height=args.half_height,
        nu=args.kinematic_viscosity,
        point_count=args.points,
        max_iterations=args.max_iterations,
        relaxation=args.relaxation,
        tolerance=args.tolerance,
        initial_mut_ratio=args.initial_mut_ratio,
        max_nu_tilde_ratio=max_ratio,
    )
    write_profile_csv(args.output, profile)
    metadata_path = args.output.with_suffix(".json")
    write_metadata(metadata_path, args, profile)

    print("Wrote {}".format(args.output))
    print("Wrote {}".format(metadata_path))
    print("Integrated bulk velocity: {:.10e} m/s".format(float(profile["bulk_velocity"])))
    print("Re_H: {:.6g}".format(args.bulk_velocity * args.half_height / args.kinematic_viscosity))
    print("Iterations: {}, residual: {:.3e}".format(profile["iterations"], profile["residual"]))
    print(
        "nu_tilde range: [{:.3e}, {:.3e}]".format(
            float(np.min(profile["nu_tilde"])),  # type: ignore[arg-type]
            float(np.max(profile["nu_tilde"])),  # type: ignore[arg-type]
        )
    )
    print(
        "nu_t/nu range: [{:.3e}, {:.3e}]".format(
            float(np.min(profile["nu_t_over_nu"])),  # type: ignore[arg-type]
            float(np.max(profile["nu_t_over_nu"])),  # type: ignore[arg-type]
        )
    )


if __name__ == "__main__":
    main()
