from dolfin import (
    MPI,
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

from Utilities_SharedTO import (
    as_list,
)


def nu_tilde_from_viscosity_ratio(ratio, nu_lam, Cv1=7.1):
    """Compute the SA working variable nu_tilde from a target turbulent viscosity ratio.

    The SA constitutive relation is:
        nu_t = nu_tilde * fv1(chi),   chi = nu_tilde / nu_lam
        fv1  = chi^3 / (chi^3 + Cv1^3)

    So we solve  f(chi) = chi * fv1(chi) - ratio = 0  by Newton iteration,
    then return  nu_tilde = chi * nu_lam.

    Args:
        ratio:   target nu_t / nu_lam  (e.g. 5.0 for moderately turbulent internal flow)
        nu_lam:  laminar kinematic viscosity  mu / rho
        Cv1:     SA model constant (default 7.1)

    Typical values for internal flows:
        ratio = 1-3   low turbulence (near-laminar)
        ratio = 5-10  moderate turbulence (Re ~ 1000-5000)
        ratio > 20    fully developed turbulent pipe flow
    """
    ratio = float(ratio)
    Cv1_3 = Cv1 ** 3
    chi = max(ratio, 1.0)  # initial guess: high-chi limit gives chi ≈ ratio
    for _ in range(50):
        chi3 = chi ** 3
        fv1 = chi3 / (chi3 + Cv1_3)
        f = chi * fv1 - ratio
        dfv1_dchi = 3.0 * chi ** 2 * Cv1_3 / (chi3 + Cv1_3) ** 2
        df = fv1 + chi * dfv1_dchi
        chi -= f / df
        chi = max(chi, 1.0e-10)
        if abs(f) < 1.0e-13 * ratio:
            break
    return chi * float(nu_lam)


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
    solver.parameters["newton_solver"]["report"] = False
    solver.solve()
    return y


def build_direct_wall_distance_solver(
    space,
    boundaries_data,
    wall_markers,
    custom_dx,
    fluid_density_indicator,
    relaxation,
    alpha_y_value,
    n_y_value,
    eikonal_eps=DOLFIN_EPS,
    newton_rtol=1.0e-8,
    newton_atol=1.0e-10,
    newton_max_iters=80,
    newton_relax=0.5,
    penalty_homotopy=None,
    solid_guess_weight=1.0,
    extra_relaxations=None,
    solid_threshold=1.0,
    initial_wall_distance=None,
):
    """
    Solve a penalized direct-y wall-distance equation:

        sqrt(|grad(y)|^2 + eps) - 1 - eta * Delta(y) + alpha_y * chi_s^n * y = 0

    with y = 0 on the physical no-slip walls.

    Returns:
    - wall_distance: Function holding the current penalized direct-y field
    - update_wall_distance: callback to re-solve y for the current design
    """
    relaxation_constant = Constant(relaxation)
    alpha_y_constant = Constant(alpha_y_value)
    n_y_constant = Constant(n_y_value)
    solid_threshold_value = float(solid_threshold)
    solid_threshold_constant = Constant(solid_threshold_value)
    eikonal_eps_constant = Constant(eikonal_eps)

    wall_bc = [
        DirichletBC(space, Constant(0.0), boundaries_data, marker)
        for marker in as_list(wall_markers)
    ]

    if initial_wall_distance is None:
        wall_distance = calculate_distance_field(
            space,
            boundaries_data,
            wall_markers,
            custom_dx,
            relaxation=relaxation,
        )
    elif isinstance(initial_wall_distance, Function):
        wall_distance = Function(space)
        wall_distance.assign(initial_wall_distance)
    else:
        wall_distance = project(initial_wall_distance, space)
    enforce_scalar_floor(wall_distance, 0.0)

    z = TestFunction(space)
    if 0.0 < solid_threshold_value < 1.0:
        solid_indicator = positive_part(solid_threshold_constant - fluid_density_indicator) / solid_threshold_constant
    else:
        solid_indicator = positive_part(Constant(1.0) - fluid_density_indicator)
    solid_penalty_indicator = solid_indicator**n_y_constant

    F = (
        sqrt(inner(grad(wall_distance), grad(wall_distance)) + eikonal_eps_constant) * z * custom_dx
        - Constant(1.0) * z * custom_dx
        + relaxation_constant * inner(grad(wall_distance), grad(z)) * custom_dx
        + alpha_y_constant * solid_penalty_indicator * wall_distance * z * custom_dx
    )
    problem = NonlinearVariationalProblem(
        F,
        wall_distance,
        bcs=wall_bc,
        J=derivative(F, wall_distance),
    )
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["report"] = False
    solver.parameters["newton_solver"]["relative_tolerance"] = float(newton_rtol)
    solver.parameters["newton_solver"]["absolute_tolerance"] = float(newton_atol)
    solver.parameters["newton_solver"]["maximum_iterations"] = int(newton_max_iters)
    solver.parameters["newton_solver"]["relaxation_parameter"] = float(newton_relax)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = True

    homotopy_scales = []
    if penalty_homotopy is None:
        penalty_homotopy = (0.0, 0.1, 0.25, 0.5, 1.0)
    if isinstance(penalty_homotopy, np.ndarray):
        penalty_homotopy = penalty_homotopy.tolist()
    elif not isinstance(penalty_homotopy, (list, tuple)):
        penalty_homotopy = [penalty_homotopy]
    for scale in penalty_homotopy:
        scale = float(scale)
        if 0.0 <= scale <= 1.0 and scale not in homotopy_scales:
            homotopy_scales.append(scale)
    if alpha_y_value > 0.0:
        if not homotopy_scales:
            homotopy_scales = [0.0, 1.0]
        if homotopy_scales[0] != 0.0:
            homotopy_scales.insert(0, 0.0)
        if homotopy_scales[-1] != 1.0:
            homotopy_scales.append(1.0)
    else:
        homotopy_scales = [1.0]

    relaxation_candidates = []
    raw_relaxations = [
        float(newton_relax),
        0.5 * float(newton_relax),
        0.25 * float(newton_relax),
        0.1,
        0.05,
        0.02,
        0.01,
    ]
    if extra_relaxations is not None:
        if isinstance(extra_relaxations, np.ndarray):
            extra_relaxations = extra_relaxations.tolist()
        elif not isinstance(extra_relaxations, (list, tuple)):
            extra_relaxations = [extra_relaxations]
        raw_relaxations.extend(float(candidate) for candidate in extra_relaxations)
    for candidate in raw_relaxations:
        if candidate > 0.0 and candidate not in relaxation_candidates:
            relaxation_candidates.append(candidate)

    initial_solid_guess_weight = min(max(float(solid_guess_weight), 0.0), 1.0)
    has_successful_update = False

    def update_wall_distance():
        nonlocal has_successful_update
        previous_values = wall_distance.vector().get_local().copy()
        last_error = None
        if (not has_successful_update) and initial_solid_guess_weight > 0.0:
            solid_blend = Constant(initial_solid_guess_weight) * solid_indicator
            warm_start = project(
                (Constant(1.0) - solid_blend) * wall_distance,
                space,
            )
            wall_distance.assign(warm_start)
            enforce_scalar_floor(wall_distance, 0.0)
        else:
            wall_distance.vector().set_local(previous_values)
            wall_distance.vector().apply("insert")

        scale_sequences = []
        if has_successful_update:
            scale_sequences.append([1.0])
        if (not has_successful_update) or len(homotopy_scales) > 1:
            scale_sequences.append(list(homotopy_scales))
        if not scale_sequences:
            scale_sequences = [[1.0]]

        for sequence_idx, scales in enumerate(scale_sequences):
            if sequence_idx > 0:
                wall_distance.vector().set_local(previous_values)
                wall_distance.vector().apply("insert")
            sequence_solved = True
            for scale in scales:
                alpha_y_constant.assign(float(scale) * float(alpha_y_value))
                stage_start_values = wall_distance.vector().get_local().copy()
                stage_solved = False
                for candidate in relaxation_candidates:
                    wall_distance.vector().set_local(stage_start_values)
                    wall_distance.vector().apply("insert")
                    solver.parameters["newton_solver"]["relaxation_parameter"] = candidate
                    try:
                        solver.solve()
                        enforce_scalar_floor(wall_distance, 0.0)
                        stage_solved = True
                        break
                    except RuntimeError as err:
                        last_error = err
                if not stage_solved:
                    sequence_solved = False
                    break
            if sequence_solved:
                alpha_y_constant.assign(float(alpha_y_value))
                has_successful_update = True
                return

        wall_distance.vector().set_local(previous_values)
        wall_distance.vector().apply("insert")
        alpha_y_constant.assign(float(alpha_y_value))
        if MPI.rank(space.mesh().mpi_comm()) == 0:
            print(
                "Warning: direct-y wall-distance update did not converge; reusing previous wall-distance field."
            )
            if last_error is not None:
                print("  Last wall-distance solve error: {}".format(last_error))

    return wall_distance, update_wall_distance


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
    newton_relax=0.5,
    penalty_homotopy=None,
    solid_guess_weight=1.0,
    extra_relaxations=None,
    solid_threshold=1.0,
    initial_wall_distance=None,
    prefer_pseudo_time=False,
):
    """
    Penalized reciprocal wall-distance equation (Yoon 2016 Eq. 25) in terms of G.

    Returns:
    - wall_distance: UFL expression for y reconstructed from G
    - update_reciprocal_distance: callback to re-solve G for the current design
    """
    g0_constant = Constant(g0_value)
    sigma_w_constant = Constant(sigma_w)
    alpha_g_constant = Constant(alpha_g_value)
    n_g_constant = Constant(n_g_value)
    solid_threshold_value = float(solid_threshold)
    solid_threshold_constant = Constant(solid_threshold_value)

    wall_bc = [
        DirichletBC(space, g0_constant, boundaries_data, marker)
        for marker in as_list(wall_markers)
    ]

    # Robust initialization from either a config-provided wall-distance field
    # or the standard geometric wall-distance solve.
    if initial_wall_distance is None:
        y_initial = calculate_distance_field(
            space,
            boundaries_data,
            wall_markers,
            custom_dx,
            relaxation=sigma_w,
        )
    elif isinstance(initial_wall_distance, Function):
        y_initial = Function(space)
        y_initial.assign(initial_wall_distance)
    else:
        y_initial = project(initial_wall_distance, space)

    # Yoon 2016 Eq. (15): l_w = 1/G - 1/G0  ->  G = 1 / (l_w + 1/G0)
    reciprocal_distance = project(
        Constant(1.0) / (y_initial + Constant(1.0 / g0_value)),
        space,
    )
    enforce_scalar_floor(reciprocal_distance, g_floor_value)

    z = TestFunction(space)
    # TO convention in this code: fluid_density_indicator = 1 in fluid, 0 in solid.
    if 0.0 < solid_threshold_value < 1.0:
        solid_indicator = positive_part(solid_threshold_constant - fluid_density_indicator) / solid_threshold_constant
    else:
        solid_indicator = positive_part(Constant(1.0) - fluid_density_indicator)
    solid_penalty_indicator = solid_indicator**n_g_constant
    penalty = alpha_g_constant * (reciprocal_distance - g0_constant) * solid_penalty_indicator

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
    solver.parameters["newton_solver"]["report"] = False
    solver.parameters["newton_solver"]["relative_tolerance"] = float(newton_rtol)
    solver.parameters["newton_solver"]["absolute_tolerance"] = float(newton_atol)
    solver.parameters["newton_solver"]["maximum_iterations"] = int(newton_max_iters)
    solver.parameters["newton_solver"]["relaxation_parameter"] = float(newton_relax)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = True

    wall_distance = positive_part(
        Constant(1.0) / (reciprocal_distance + Constant(g_floor_value))
        - Constant(1.0 / g0_value)
    )

    homotopy_scales = []
    if penalty_homotopy is None:
        penalty_homotopy = (0.0, 0.1, 0.25, 0.5, 1.0)
    if isinstance(penalty_homotopy, np.ndarray):
        penalty_homotopy = penalty_homotopy.tolist()
    elif not isinstance(penalty_homotopy, (list, tuple)):
        penalty_homotopy = [penalty_homotopy]
    for scale in penalty_homotopy:
        scale = float(scale)
        if 0.0 <= scale <= 1.0 and scale not in homotopy_scales:
            homotopy_scales.append(scale)
    if alpha_g_value > 0.0:
        if not homotopy_scales:
            homotopy_scales = [0.0, 1.0]
        if homotopy_scales[0] != 0.0:
            homotopy_scales.insert(0, 0.0)
        if homotopy_scales[-1] != 1.0:
            homotopy_scales.append(1.0)
    else:
        homotopy_scales = [1.0]

    relaxation_candidates = []
    raw_relaxations = [
        float(newton_relax),
        0.5 * float(newton_relax),
        0.25 * float(newton_relax),
        0.1,
        0.05,
        0.02,
        0.01,
    ]
    if extra_relaxations is not None:
        if isinstance(extra_relaxations, np.ndarray):
            extra_relaxations = extra_relaxations.tolist()
        elif not isinstance(extra_relaxations, (list, tuple)):
            extra_relaxations = [extra_relaxations]
        raw_relaxations.extend(float(candidate) for candidate in extra_relaxations)
    for candidate in raw_relaxations:
        if candidate > 0.0 and candidate not in relaxation_candidates:
            relaxation_candidates.append(candidate)

    initial_solid_guess_weight = min(max(float(solid_guess_weight), 0.0), 1.0)
    pseudo_dt_candidates = [0.02, 0.01, 0.005]
    pseudo_max_steps = 80
    pseudo_relaxation = 0.5
    pseudo_rtol = 1.0e-4
    reciprocal_distance_previous = Function(space)
    g_trial = TrialFunction(space)
    prefer_pseudo_time = bool(prefer_pseudo_time)
    has_successful_update = False

    def solve_with_pseudo_time(scale):
        alpha_stage_constant = Constant(float(scale) * float(alpha_g_value))
        stage_start_values = reciprocal_distance.vector().get_local().copy()
        last_error = None
        last_finite_values = None
        best_change_norm = None

        for pseudo_dt in pseudo_dt_candidates:
            reciprocal_distance.vector().set_local(stage_start_values)
            reciprocal_distance.vector().apply("insert")
            enforce_scalar_floor(reciprocal_distance, g_floor_value)

            for _ in range(pseudo_max_steps):
                previous_step_values = reciprocal_distance.vector().get_local().copy()
                reciprocal_distance_previous.assign(reciprocal_distance)

                reaction_coefficient = (
                    (Constant(1.0) + Constant(2.0) * sigma_w_constant) * reciprocal_distance_previous**3
                    + alpha_stage_constant * solid_penalty_indicator
                )
                a = (
                    Constant(1.0 / pseudo_dt) * g_trial * z * custom_dx
                    + (Constant(1.0) - sigma_w_constant)
                    * inner(grad(reciprocal_distance_previous), grad(g_trial))
                    * z
                    * custom_dx
                    - sigma_w_constant
                    * reciprocal_distance_previous
                    * inner(grad(g_trial), grad(z))
                    * custom_dx
                    - reaction_coefficient * g_trial * z * custom_dx
                )
                L = (
                    Constant(1.0 / pseudo_dt) * reciprocal_distance_previous * z * custom_dx
                    - alpha_stage_constant * solid_penalty_indicator * g0_constant * z * custom_dx
                )

                try:
                    solve(a == L, reciprocal_distance, wall_bc)
                except RuntimeError as err:
                    last_error = err
                    break

                enforce_scalar_floor(reciprocal_distance, g_floor_value)

                if pseudo_relaxation < 1.0:
                    updated_values = reciprocal_distance.vector().get_local()
                    blended_values = (
                        pseudo_relaxation * updated_values
                        + (1.0 - pseudo_relaxation) * previous_step_values
                    )
                    reciprocal_distance.vector().set_local(blended_values)
                    reciprocal_distance.vector().apply("insert")
                    enforce_scalar_floor(reciprocal_distance, g_floor_value)

                current_values = reciprocal_distance.vector().get_local()
                if not np.all(np.isfinite(current_values)):
                    last_error = RuntimeError("Pseudo-time wall-distance update produced non-finite values.")
                    break
                last_finite_values = current_values.copy()

                scale_norm = max(
                    np.max(np.abs(previous_step_values)),
                    np.max(np.abs(current_values)),
                    abs(float(g0_value)),
                    1.0,
                )
                change_norm = np.max(np.abs(current_values - previous_step_values))
                if best_change_norm is None or change_norm < best_change_norm:
                    best_change_norm = change_norm
                if change_norm <= pseudo_rtol * scale_norm:
                    return True, last_error

        if last_finite_values is not None:
            reciprocal_distance.vector().set_local(last_finite_values)
            reciprocal_distance.vector().apply("insert")
            enforce_scalar_floor(reciprocal_distance, g_floor_value)
            return True, last_error

        reciprocal_distance.vector().set_local(stage_start_values)
        reciprocal_distance.vector().apply("insert")
        enforce_scalar_floor(reciprocal_distance, g_floor_value)
        return False, last_error

    def update_reciprocal_distance():
        nonlocal has_successful_update
        previous_values = reciprocal_distance.vector().get_local().copy()
        last_error = None
        if (not has_successful_update) and initial_solid_guess_weight > 0.0:
            solid_blend = Constant(initial_solid_guess_weight) * solid_indicator
            warm_start = project(
                reciprocal_distance + solid_blend * (g0_constant - reciprocal_distance),
                space,
            )
            reciprocal_distance.assign(warm_start)
            enforce_scalar_floor(reciprocal_distance, g_floor_value)
        else:
            reciprocal_distance.vector().set_local(previous_values)
            reciprocal_distance.vector().apply("insert")

        scale_sequences = []
        if has_successful_update:
            scale_sequences.append([1.0])
        if (not has_successful_update) or len(homotopy_scales) > 1:
            scale_sequences.append(list(homotopy_scales))
        if not scale_sequences:
            scale_sequences = [[1.0]]

        for sequence_idx, scales in enumerate(scale_sequences):
            if sequence_idx > 0:
                reciprocal_distance.vector().set_local(previous_values)
                reciprocal_distance.vector().apply("insert")
            sequence_solved = True
            for scale in scales:
                alpha_g_constant.assign(float(scale) * float(alpha_g_value))
                stage_start_values = reciprocal_distance.vector().get_local().copy()
                stage_solved = False
                if not prefer_pseudo_time:
                    for candidate in relaxation_candidates:
                        reciprocal_distance.vector().set_local(stage_start_values)
                        reciprocal_distance.vector().apply("insert")
                        solver.parameters["newton_solver"]["relaxation_parameter"] = candidate
                        try:
                            solver.solve()
                            enforce_scalar_floor(reciprocal_distance, g_floor_value)
                            stage_solved = True
                            break
                        except RuntimeError as err:
                            last_error = err
                if not stage_solved:
                    reciprocal_distance.vector().set_local(stage_start_values)
                    reciprocal_distance.vector().apply("insert")
                    enforce_scalar_floor(reciprocal_distance, g_floor_value)
                    stage_solved, last_error = solve_with_pseudo_time(scale)
                    if not stage_solved:
                        sequence_solved = False
                        break
            if sequence_solved:
                alpha_g_constant.assign(float(alpha_g_value))
                has_successful_update = True
                return

        reciprocal_distance.vector().set_local(previous_values)
        reciprocal_distance.vector().apply("insert")
        alpha_g_constant.assign(float(alpha_g_value))
        if MPI.rank(space.mesh().mpi_comm()) == 0:
            print(
                "Warning: penalized wall-distance update did not converge; reusing previous wall-distance field."
            )
            if last_error is not None:
                print("  Last wall-distance solve error: {}".format(last_error))

    return wall_distance, update_reciprocal_distance
