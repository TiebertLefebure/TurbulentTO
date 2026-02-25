from dolfin import (
    Constant,
    DOLFIN_EPS,
    DirichletBC,
    Function,
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

from Utilities_LaminarTO import (
    as_list,
)


def positive_part(expr):
    zero = Constant(0.0)
    return conditional(gt(expr, zero), expr, zero)


def enforce_scalar_floor(scalar_function, floor_value):
    """Enforce scalar_function >= floor_value in-place."""
    values = scalar_function.vector().get_local()
    values = np.maximum(values, floor_value)
    scalar_function.vector().set_local(values)
    scalar_function.vector().apply("insert")


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
    """
    Penalized reciprocal wall-distance equation (Yoon 2016 Eq. 25) in terms of G.

    Returns:
    - wall_distance: UFL expression for y reconstructed from G
    - reciprocal_distance: Function G
    - update_reciprocal_distance: callback to re-solve G for the current design
    """
    g0_constant = Constant(g0_value)
    sigma_w_constant = Constant(sigma_w)
    alpha_g_constant = Constant(alpha_g_value)
    n_g_constant = Constant(n_g_value)

    wall_bc = [
        DirichletBC(space, g0_constant, boundaries_data, marker)
        for marker in as_list(wall_markers)
    ]

    # Robust initialization from the standard wall-distance field.
    y_initial = calculate_distance_field(
        space,
        boundaries_data,
        wall_markers,
        custom_dx,
        relaxation=sigma_w,
    )

    # Yoon 2016 Eq. (15): l_w = 1/G - 1/G0  ->  G = 1 / (l_w + 1/G0)
    reciprocal_distance = project(
        Constant(1.0) / (y_initial + Constant(1.0 / g0_value)),
        space,
    )
    enforce_scalar_floor(reciprocal_distance, g_floor_value)

    z = TestFunction(space)
    # TO convention in this code: fluid_density_indicator = 1 in fluid, 0 in solid.
    solid_indicator = positive_part(Constant(1.0) - fluid_density_indicator)
    penalty = alpha_g_constant * (reciprocal_distance - g0_constant) * solid_indicator**n_g_constant

    # Yoon 2016 Eq. (25), weak form for reciprocal wall distance G.
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
