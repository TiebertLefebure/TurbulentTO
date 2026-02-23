import argparse
import importlib
import os
import shutil

from dolfin import (
    Constant,
    DOLFIN_EPS,
    DirichletBC,
    Function,
    MPI,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    TestFunction,
    TrialFunction,
    conditional,
    derivative,
    grad,
    gt,
    inner,
    lhs,
    project,
    rhs,
    solve,
    sqrt,
)
import numpy as np


def normalize_module_name(module_name):
    normalized = module_name.strip()
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    normalized = normalized.replace(os.sep, ".")
    return normalized


def load_config_module_from_cli():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        required=True,
        help="Python module name for solver configuration (required).",
    )
    args, _unknown = parser.parse_known_args()
    module_name = normalize_module_name(args.config)
    if not module_name:
        raise ValueError("Empty config module name is not allowed.")
    return module_name, importlib.import_module(module_name)


def float_scalar(value):
    if hasattr(value, "values"):
        values = value.values()
        if len(values) == 1:
            return float(values[0])
    return float(value)


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def match_count(values, target_size):
    values = as_list(values)
    if len(values) == target_size:
        return values
    if len(values) == 1:
        return values * target_size
    if len(values) < target_size:
        return values + [values[-1]] * (target_size - len(values))
    return values[:target_size]


def positive_part(expr):
    zero = Constant(0.0)
    return conditional(gt(expr, zero), expr, zero)


def enforce_scalar_floor(scalar_function, floor_value):
    """Enforce scalar_function >= floor_value in-place."""
    values = scalar_function.vector().get_local()
    values = np.maximum(values, floor_value)
    scalar_function.vector().set_local(values)
    scalar_function.vector().apply("insert")


def ensure_clean_dir(path, comm=MPI.comm_world):
    # Avoid MPI races where multiple ranks delete/create the same folder.
    if MPI.rank(comm) == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    MPI.barrier(comm)


def calculate_distance_field(space, boundaries_data, wall_markers, custom_dx, relaxation=0.01):
    """Compute distance-to-wall field y with y=0 on no-slip walls."""
    wall_bc = [
        DirichletBC(space, Constant(0.0), boundaries_data, marker)
        for marker in as_list(wall_markers)
    ]

    y = Function(space)
    dy = TrialFunction(space)
    z = TestFunction(space)

    # Linear initialization (Poisson problem with wall BC y=0)
    linear_problem = inner(grad(dy), grad(z)) * custom_dx - Constant(1.0) * z * custom_dx
    solve(lhs(linear_problem) == rhs(linear_problem), y, wall_bc)

    # Smoothed Eikonal solve (Laplacian relaxation)
    F = (
        sqrt(inner(grad(y), grad(y)) + DOLFIN_EPS) * z * custom_dx
        - Constant(1.0) * z * custom_dx
        + Constant(relaxation) * inner(grad(y), grad(z)) * custom_dx
    )
    problem = NonlinearVariationalProblem(F, y, bcs=wall_bc, J=derivative(F, y))
    solver = NonlinearVariationalSolver(problem)
    solver.solve()
    return y


def build_penalized_wall_distance_solver(
    space,
    boundaries_data,
    wall_markers,
    custom_dx,
    fluid_density_indicator,
    sigma_w,
    g0_value,
    alpha_g_value,
    n_g_value,
    g_floor_value,
    newton_rtol=1.0e-8,
    newton_atol=1.0e-10,
    newton_max_iters=80,
    newton_relax=1.0,
):
    g0_constant = Constant(g0_value)
    sigma_w_constant = Constant(sigma_w)
    alpha_g_constant = Constant(alpha_g_value)
    n_g_constant = Constant(n_g_value)

    wall_bc = [
        DirichletBC(space, g0_constant, boundaries_data, marker)
        for marker in as_list(wall_markers)
    ]

    # Use the classic wall-distance as a robust initial guess for reciprocal distance G.
    y_initial = calculate_distance_field(
        space,
        boundaries_data,
        wall_markers,
        custom_dx,
        relaxation=sigma_w,
    )

    # Yoon 2016 Eq.(15)
    # l_w = 1/G - 1/G0  ->  G = 1 / (l_w + 1/G0)
    reciprocal_distance = project(
        Constant(1.0) / (y_initial + Constant(1.0 / g0_value)),
        space,
    )
    enforce_scalar_floor(reciprocal_distance, g_floor_value)

    z = TestFunction(space)
    # In this TO code, rho_effective=1 means fluid and rho_effective=0 means solid.
    solid_indicator = positive_part(Constant(1.0) - fluid_density_indicator)
    penalty = alpha_g_constant * (reciprocal_distance - g0_constant) * solid_indicator**n_g_constant

    # Weak form: grad(G).grad(G) + sigma_w * G * Laplacian(G) = (1 + 2*sigma_w)*G^4 + penalty
    F = (
        (Constant(1.0) - sigma_w_constant)
        * inner(grad(reciprocal_distance), grad(reciprocal_distance))
        * z
        * custom_dx
        - sigma_w_constant
        * reciprocal_distance
        * inner(grad(reciprocal_distance), grad(z))
        * custom_dx
        - ((Constant(1.0) + Constant(2.0) * sigma_w_constant) * reciprocal_distance**4 + penalty)
        * z
        * custom_dx
    )
    problem = NonlinearVariationalProblem(
        F,
        reciprocal_distance,
        bcs=wall_bc,
        J=derivative(F, reciprocal_distance),
    )
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["relative_tolerance"] = float(newton_rtol)
    solver.parameters["newton_solver"]["absolute_tolerance"] = float(newton_atol)
    solver.parameters["newton_solver"]["maximum_iterations"] = int(newton_max_iters)
    solver.parameters["newton_solver"]["relaxation_parameter"] = float(newton_relax)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = True

    wall_distance = positive_part(
        Constant(1.0) / (reciprocal_distance + Constant(g_floor_value))
        - Constant(1.0 / g0_value)
    )

    def update_reciprocal_distance():
        solver.solve()
        enforce_scalar_floor(reciprocal_distance, g_floor_value)

    return wall_distance, reciprocal_distance, update_reciprocal_distance
