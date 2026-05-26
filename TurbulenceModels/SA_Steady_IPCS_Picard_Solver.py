"""Generic steady RANS-SA solver driver.

Case scripts build the mesh, spaces, boundary conditions, IPCS forms, and SA
model object. This file only runs the coupled iteration process.
"""

from dolfin import *
from mpi4py import MPI as MPI4PY
import numpy as np
import os
import time

from Utilities import (
    bound_from_bellow,
    load_h5_file,
    save_h5_file,
    save_list,
    save_pvd_file,
    terminal_print,
    visualize_convergence,
    visualize_functions,
)


def root_print(message, is_root=True):
    if is_root:
        print(message)


def l2_norm_diff(f1, f0, dx_measure):
    diff = f1 - f0
    if f1.ufl_shape == ():
        return np.sqrt(float(assemble(diff**2 * dx_measure)))
    return np.sqrt(float(assemble(dot(diff, diff) * dx_measure)))


def relative_vector_diff(f1, f0):
    f1_local = f1.vector().get_local()
    f0_local = f0.vector().get_local()
    diff_local = f1_local - f0_local

    diff_sq_local = float(np.dot(diff_local, diff_local))
    norm_sq_local = float(np.dot(f1_local, f1_local))
    diff_sq = MPI4PY.COMM_WORLD.allreduce(diff_sq_local, op=MPI4PY.SUM)
    norm_sq = MPI4PY.COMM_WORLD.allreduce(norm_sq_local, op=MPI4PY.SUM)
    return np.sqrt(diff_sq) / max(np.sqrt(norm_sq), 1.0e-12)


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return list(value)
    return [value]


def solve_linear_system(A, x, b, solver_name, preconditioner_name=None):
    if solver_name in (None, "", "default"):
        if preconditioner_name in (None, "", "default"):
            solve(A, x, b)
        else:
            solve(A, x, b, "default", preconditioner_name)
    else:
        if preconditioner_name in (None, "", "default"):
            solve(A, x, b, solver_name)
        else:
            solve(A, x, b, solver_name, preconditioner_name)


def run_steady_sa_ipcs_picard(
    *,
    simulation_prm,
    post_processing,
    saving_directory,
    dx,
    dt,
    a_1,
    l_1,
    a_2,
    l_2,
    a_3,
    l_3,
    bcu,
    bcp,
    u0,
    u1,
    p0,
    p1,
    velocity_space,
    pressure_space,
    turbulence_space,
    turbulence_model,
    normalize_pressure_mean=False,
    ds=None,
    pressure_drop_metric=None,
    picard_diagnostics=None,
    picard_convergence=None,
    is_root=True,
):
    """Run steady RANS-SA using an outer Picard loop and inner pseudo-time IPCS solves."""
    default_flow_solver = simulation_prm.get(
        "FLOW_IPCS_LINEAR_SOLVER",
        simulation_prm.get("NS_LINEAR_SOLVER", simulation_prm.get("LINEAR_SOLVER", "mumps")),
    )
    default_flow_preconditioner = simulation_prm.get(
        "FLOW_IPCS_LINEAR_PRECONDITIONER",
        simulation_prm.get("NS_LINEAR_PRECONDITIONER", simulation_prm.get("LINEAR_PRECONDITIONER", None)),
    )
    velocity_solver = simulation_prm.get("FLOW_IPCS_VELOCITY_LINEAR_SOLVER", default_flow_solver)
    velocity_preconditioner = simulation_prm.get(
        "FLOW_IPCS_VELOCITY_LINEAR_PRECONDITIONER",
        default_flow_preconditioner,
    )
    pressure_solver = simulation_prm.get("FLOW_IPCS_PRESSURE_LINEAR_SOLVER", default_flow_solver)
    pressure_preconditioner = simulation_prm.get(
        "FLOW_IPCS_PRESSURE_LINEAR_PRECONDITIONER",
        default_flow_preconditioner,
    )
    correction_solver = simulation_prm.get("FLOW_IPCS_CORRECTION_LINEAR_SOLVER", default_flow_solver)
    correction_preconditioner = simulation_prm.get(
        "FLOW_IPCS_CORRECTION_LINEAR_PRECONDITIONER",
        default_flow_preconditioner,
    )
    sa_relaxation = float(simulation_prm.get(
        "COUPLED_PICARD_SA_RELAXATION",
        simulation_prm.get("TURBULENCE_RELAXATION", simulation_prm.get("NUT_RELAXATION_FACTOR", 1.0)),
    ))
    u_relaxation = float(simulation_prm.get(
        "FLOW_IPCS_VELOCITY_RELAXATION",
        simulation_prm.get("FORWARD_IPCS_VEL_RELAXATION", simulation_prm.get("U_RELAXATION_FACTOR", 1.0)),
    ))
    p_relaxation = float(simulation_prm.get(
        "FLOW_IPCS_PRESSURE_RELAXATION",
        simulation_prm.get("FORWARD_IPCS_P_RELAXATION", simulation_prm.get("P_RELAXATION_FACTOR", u_relaxation)),
    ))

    max_picard = int(simulation_prm.get(
        "COUPLED_PICARD_MAX_STEPS",
        simulation_prm.get(
            "STEADY_MAX_PICARD_ITERATIONS",
            simulation_prm.get("MAX_PICARD_ITERATIONS", simulation_prm.get("MAX_ITERATIONS", 100)),
        ),
    ))
    flow_max_iters = int(simulation_prm.get(
        "FLOW_IPCS_MAX_STEPS",
        simulation_prm.get("FORWARD_IPCS_MAX_ITERS", simulation_prm.get("FLOW_MAX_ITERATIONS", min(max_picard, 500))),
    ))
    final_flow_max_iters = int(simulation_prm.get("FLOW_IPCS_FINAL_MAX_STEPS", flow_max_iters))
    sa_sweeps = max(1, int(simulation_prm.get(
        "COUPLED_PICARD_SA_SWEEPS_PER_STEP",
        simulation_prm.get("STEADY_SA_SWEEPS", 1),
    )))
    nu_tilde_floor = simulation_prm.get("SA_NU_TILDE_FLOOR", None)
    picard_checkpoint_every = int(simulation_prm.get("PICARD_CHECKPOINT_EVERY", 0))
    checkpoint_require_flow_convergence = bool(simulation_prm.get(
        "PICARD_CHECKPOINT_REQUIRE_FLOW_CONVERGENCE",
        True,
    ))
    checkpoint_h5_dir = simulation_prm.get(
        "PICARD_CHECKPOINT_H5_DIRECTORY",
        simulation_prm.get("RESTART_H5_DIRECTORY", saving_directory.get("H5_FILES")),
    )
    checkpoint_save_pvd = bool(simulation_prm.get("PICARD_CHECKPOINT_SAVE_PVD", False))
    checkpoint_pvd_dir = simulation_prm.get(
        "PICARD_CHECKPOINT_PVD_DIRECTORY",
        saving_directory.get("PVD_FILES"),
    )

    tolerance_global = float(simulation_prm.get("COUPLED_PICARD_TOLERANCE", simulation_prm.get("TOLERANCE", 1.0e-6)))
    tolerance_u = float(simulation_prm.get(
        "COUPLED_PICARD_VELOCITY_TOLERANCE",
        simulation_prm.get("TOLERANCE_U", tolerance_global),
    ))
    tolerance_p = float(simulation_prm.get(
        "COUPLED_PICARD_PRESSURE_TOLERANCE",
        simulation_prm.get("TOLERANCE_P", tolerance_global),
    ))
    tolerance_nu_tilde = float(simulation_prm.get(
        "COUPLED_PICARD_NU_TILDE_TOLERANCE",
        simulation_prm.get("TOLERANCE_NU_TILDE", tolerance_global),
    ))
    flow_tolerance_u = float(simulation_prm.get(
        "FLOW_IPCS_VELOCITY_TOLERANCE",
        simulation_prm.get("FORWARD_IPCS_VELOCITY_RTOL", tolerance_u),
    ))
    flow_tolerance_p = float(simulation_prm.get(
        "FLOW_IPCS_PRESSURE_TOLERANCE",
        simulation_prm.get("FORWARD_IPCS_PRESSURE_RTOL", tolerance_p),
    ))

    dt.assign(float(simulation_prm.get("FLOW_IPCS_TIME_STEP", simulation_prm.get("FORWARD_IPCS_DT"))))
    log_every = max(1, int(simulation_prm.get("FLOW_IPCS_LOG_EVERY", simulation_prm.get("FORWARD_IPCS_LOG_EVERY", 25))))
    log_flow_iterations = bool(simulation_prm.get(
        "FLOW_IPCS_VERBOSE",
        simulation_prm.get("FORWARD_IPCS_VERBOSE", False),
    ))

    def scalar_value(value):
        if value is None:
            return None
        return float(value)

    def build_pressure_drop_evaluator():
        if not pressure_drop_metric:
            return None, None, None, None

        pressure_scale = scalar_value(pressure_drop_metric.get(
            "PRESSURE_SCALE",
            pressure_drop_metric.get("PRESSURE_TO_PA", 1.0),
        ))
        if pressure_scale is None:
            pressure_scale = 1.0
        pressure_unit = pressure_drop_metric.get("UNIT", "Pa")

        def reported_pressure_drop(value):
            return float(pressure_scale * value)

        fixed_pressure_drop = scalar_value(pressure_drop_metric.get(
            "FIXED_PRESSURE_DROP",
            pressure_drop_metric.get("PRESSURE_DROP"),
        ))
        source = pressure_drop_metric.get("SOURCE")
        if fixed_pressure_drop is not None:
            label = pressure_drop_metric.get("LABEL", "pressure drop metric")
            fixed_reported_pressure_drop = reported_pressure_drop(fixed_pressure_drop)

            def evaluate_fixed_pressure_drop(_field):
                return fixed_reported_pressure_drop

            description = "{} = {:.12e} {}".format(
                label,
                fixed_reported_pressure_drop,
                pressure_unit,
            )
            if source:
                description += " ({})".format(source)
            return evaluate_fixed_pressure_drop, label, description, pressure_unit

        if ds is None:
            raise ValueError("Pressure-drop metric requires a boundary measure 'ds'.")

        inlet_markers = as_list(pressure_drop_metric.get("INLET_MARKERS"))
        outlet_markers = as_list(pressure_drop_metric.get("OUTLET_MARKERS"))
        fixed_inlet_pressure = scalar_value(pressure_drop_metric.get("INLET_PRESSURE"))
        fixed_outlet_pressure = scalar_value(pressure_drop_metric.get("OUTLET_PRESSURE"))
        label = pressure_drop_metric.get("LABEL", "area-weighted pressure drop")

        def boundary_pressure_average(field, markers, boundary_label):
            if len(markers) == 0:
                raise ValueError(
                    "Pressure-drop metric needs {} markers when no fixed {} pressure is supplied.".format(
                        boundary_label,
                        boundary_label,
                    )
                )

            weighted_pressure = 0.0
            boundary_area = 0.0
            for marker in markers:
                marker_id = int(marker)
                marker_area = float(assemble(Constant(1.0) * ds(marker_id)))
                weighted_pressure += float(assemble(field * ds(marker_id)))
                boundary_area += marker_area

            if abs(boundary_area) <= 1.0e-30:
                raise ValueError(
                    "Pressure-drop metric found zero boundary measure for {} markers {}.".format(
                        boundary_label,
                        markers,
                    )
                )
            return weighted_pressure / boundary_area

        def pressure_average_or_fixed_value(field, markers, fixed_pressure, boundary_label):
            if fixed_pressure is not None:
                return fixed_pressure
            if len(markers) > 0:
                return boundary_pressure_average(field, markers, boundary_label)
            raise ValueError(
                "Pressure-drop metric needs {} markers or a fixed {} pressure.".format(
                    boundary_label,
                    boundary_label,
                )
            )

        def evaluate_pressure_drop(field):
            inlet_pressure = pressure_average_or_fixed_value(
                field,
                inlet_markers,
                fixed_inlet_pressure,
                "inlet",
            )
            outlet_pressure = pressure_average_or_fixed_value(
                field,
                outlet_markers,
                fixed_outlet_pressure,
                "outlet",
            )

            return reported_pressure_drop(inlet_pressure - outlet_pressure)

        inlet_description = (
            "fixed inlet pressure" if fixed_inlet_pressure is not None else "average inlet pressure"
        )
        outlet_description = (
            "fixed outlet pressure" if fixed_outlet_pressure is not None else "average outlet pressure"
        )
        description = "{} = {} - {}".format(label, inlet_description, outlet_description)
        if abs(pressure_scale - 1.0) > 1.0e-15:
            description += ", scaled by {:.6e} to {}".format(pressure_scale, pressure_unit)
        if source:
            description += " ({})".format(source)
        return evaluate_pressure_drop, label, description, pressure_unit

    (
        pressure_drop_evaluator,
        pressure_drop_label,
        pressure_drop_description,
        pressure_drop_unit,
    ) = build_pressure_drop_evaluator()

    diagnostic_specs = []
    for diagnostic in picard_diagnostics or []:
        key = diagnostic.get("key")
        evaluator = diagnostic.get("evaluator")
        if key in (None, "") or evaluator is None:
            raise ValueError("Picard diagnostics require 'key' and 'evaluator'.")
        diagnostic_specs.append({
            "key": key,
            "label": diagnostic.get("label", key),
            "unit": diagnostic.get("unit", ""),
            "format": diagnostic.get("format", "{:.6e}"),
            "evaluator": evaluator,
        })

    convergence_prm = picard_convergence or {}
    convergence_type = convergence_prm.get("type", "field_norm")
    if convergence_type not in ("field_norm", "diagnostic_window_relative_change"):
        raise ValueError(
            "Unknown Picard convergence type '{}'. Use 'field_norm' or "
            "'diagnostic_window_relative_change'.".format(convergence_type)
        )
    convergence_diagnostic_key = convergence_prm.get("diagnostic_key")
    convergence_window = int(convergence_prm.get("window", 0))
    convergence_relative_tolerance = float(convergence_prm.get("relative_tolerance", 0.0))
    convergence_value_floor = float(convergence_prm.get("value_floor", 1.0e-30))
    convergence_require_flow = bool(convergence_prm.get("require_flow_convergence", True))
    convergence_require_pressure = bool(convergence_prm.get("require_pressure_tolerance", True))
    convergence_require_nu_tilde = bool(convergence_prm.get("require_nu_tilde_tolerance", True))
    convergence_require_velocity = bool(convergence_prm.get(
        "require_velocity_tolerance",
        convergence_type == "field_norm",
    ))
    if convergence_type == "diagnostic_window_relative_change":
        diagnostic_keys = [diagnostic["key"] for diagnostic in diagnostic_specs]
        if convergence_diagnostic_key not in diagnostic_keys:
            raise ValueError(
                "Diagnostic-window Picard convergence needs diagnostic_key to match "
                "one of {}.".format(diagnostic_keys)
            )
        if convergence_window <= 0:
            raise ValueError("Diagnostic-window Picard convergence requires window > 0.")
        if convergence_relative_tolerance <= 0.0:
            raise ValueError(
                "Diagnostic-window Picard convergence requires relative_tolerance > 0."
            )

    def maybe_restart_from_saved_state():
        if not bool(simulation_prm.get("RESTART_FROM_SAVED_STATE", False)):
            return False

        restart_dir = simulation_prm.get("RESTART_H5_DIRECTORY", saving_directory.get("H5_FILES"))
        if restart_dir in (None, ""):
            message = "Restart skipped: no HDF5 restart directory configured."
            if bool(simulation_prm.get("RESTART_REQUIRE_FILES", False)):
                raise ValueError(message)
            root_print(message, is_root=is_root)
            return False

        restart_files = {
            "u": os.path.join(restart_dir, "u.h5"),
            "p": os.path.join(restart_dir, "p.h5"),
            "nu_tilde": os.path.join(restart_dir, "nu_tilde.h5"),
        }
        missing_files = [path for path in restart_files.values() if not os.path.exists(path)]
        if missing_files:
            message = "Restart skipped: missing HDF5 state file(s): {}.".format(
                ", ".join(sorted(missing_files))
            )
            if bool(simulation_prm.get("RESTART_REQUIRE_FILES", False)):
                raise FileNotFoundError(message)
            root_print(message, is_root=is_root)
            return False

        load_h5_file(u0, restart_files["u"])
        load_h5_file(p0, restart_files["p"])
        load_h5_file(turbulence_model.nu_tilde0, restart_files["nu_tilde"])

        if nu_tilde_floor is not None:
            bound_from_bellow(turbulence_model.nu_tilde0, float(nu_tilde_floor))

        for bc in bcu:
            bc.apply(u0.vector())
        for bc in bcp:
            bc.apply(p0.vector())
        turbulence_model.enforce_boundary_conditions()

        u1.assign(u0)
        for bc in bcu:
            bc.apply(u1.vector())
        p1.assign(p0)
        for bc in bcp:
            bc.apply(p1.vector())
        turbulence_model.nu_tilde1.assign(turbulence_model.nu_tilde0)
        turbulence_model.enforce_boundary_conditions()

        root_print(
            "Loaded restart state from {}.".format(restart_dir),
            is_root=is_root,
        )
        return True

    maybe_restart_from_saved_state()

    def save_h5_checkpoint_field(field, path):
        temporary_path = "{}.tmp".format(path)
        save_h5_file(field, temporary_path)
        MPI4PY.COMM_WORLD.Barrier()
        if MPI4PY.COMM_WORLD.Get_rank() == 0:
            os.replace(temporary_path, path)
        MPI4PY.COMM_WORLD.Barrier()

    def save_picard_checkpoint():
        if picard_checkpoint_every <= 0:
            return False
        if checkpoint_h5_dir in (None, ""):
            return False

        checkpoint_fields = {
            "u": u0,
            "p": p0,
            "nu_tilde": turbulence_model.nu_tilde0,
        }
        for key, field in checkpoint_fields.items():
            save_h5_checkpoint_field(field, os.path.join(checkpoint_h5_dir, key + ".h5"))
            if checkpoint_save_pvd and checkpoint_pvd_dir not in (None, ""):
                save_pvd_file(field, os.path.join(checkpoint_pvd_dir, key + ".pvd"))
        return True

    residual_keys = ["u", "p", "nu_tilde", "flow_u", "flow_p"]
    if pressure_drop_evaluator is not None:
        residual_keys.append("pressure_drop")
    for diagnostic in diagnostic_specs:
        if diagnostic["key"] in residual_keys:
            raise ValueError(
                "Picard diagnostic key '{}' conflicts with an existing residual key.".format(
                    diagnostic["key"]
                )
            )
        residual_keys.append(diagnostic["key"])
    residuals = {key: [] for key in residual_keys}
    start_time = time.time()

    u_prev_picard = Function(velocity_space)
    p_prev_picard = Function(pressure_space)
    nu_tilde_prev_picard = Function(turbulence_space)

    def normalize_pressure():
        if not normalize_pressure_mean:
            return
        domain_area = assemble(Constant(1.0) * dx)
        if abs(float(domain_area)) > 1.0e-30:
            p_average = assemble(p1 * dx) / domain_area
            p1.vector()[:] -= p_average

    pressure_null_space = None

    def apply_pressure_nullspace(A, b):
        # Body-force periodic-channel solves have no pressure Dirichlet BC.
        # Attach the constant-pressure nullspace and orthogonalize the RHS so
        # the linear pressure-correction solve has an explicit gauge treatment.
        if len(bcp) > 0:
            return

        nonlocal pressure_null_space
        if pressure_null_space is None:
            null_vector = Vector(p1.vector())
            pressure_space.dofmap().set(null_vector, 1.0)
            null_vector.apply("insert")
            null_vector *= 1.0 / null_vector.norm("l2")
            pressure_null_space = VectorSpaceBasis([null_vector])

        as_backend_type(A).set_nullspace(pressure_null_space)
        pressure_null_space.orthogonalize(b)

    def solve_flow_to_steady(label, max_iters=None, verbose=False):
        # Inner loop: keep SA viscosity fixed and advance the flow equations
        # with IPCS substeps until the velocity and pressure changes are small.
        step_limit = flow_max_iters if max_iters is None else int(max_iters)
        converged = False
        du_rel = dp_rel = np.inf

        for flow_iter in range(1, step_limit + 1):
            # IPCS step 1: tentative velocity from momentum without pressure correction.
            A_1 = assemble(a_1)
            b_1 = assemble(l_1)
            for bc in bcu:
                bc.apply(A_1, b_1)
            solve_linear_system(A_1, u1.vector(), b_1, velocity_solver, velocity_preconditioner)

            # IPCS step 2: pressure correction from the tentative velocity divergence.
            A_2 = assemble(a_2)
            b_2 = assemble(l_2)
            for bc in bcp:
                bc.apply(A_2, b_2)
            apply_pressure_nullspace(A_2, b_2)
            solve_linear_system(A_2, p1.vector(), b_2, pressure_solver, pressure_preconditioner)
            normalize_pressure()

            # IPCS step 3: velocity correction using the updated pressure field.
            A_3 = assemble(a_3)
            b_3 = assemble(l_3)
            for bc in bcu:
                bc.apply(A_3, b_3)
            solve_linear_system(A_3, u1.vector(), b_3, correction_solver, correction_preconditioner)

            du_rel = relative_vector_diff(u1, u0)
            dp_rel = relative_vector_diff(p1, p0)

            if verbose and is_root and (flow_iter == 1 or flow_iter % log_every == 0):
                terminal_print(
                    "    [{} IPCS {:04d}] du={:.3e}, dp={:.3e}".format(
                        label, flow_iter, du_rel, dp_rel
                    ),
                    is_root=is_root,
                )

            if du_rel <= flow_tolerance_u and dp_rel <= flow_tolerance_p:
                u0.assign(u1)
                p0.assign(p1)
                converged = True
                break

            # Under-relax the flow fields before the next pseudo-time IPCS step.
            u0.vector()[:] = u_relaxation * u1.vector()[:] + (1.0 - u_relaxation) * u0.vector()[:]
            p0.vector()[:] = p_relaxation * p1.vector()[:] + (1.0 - p_relaxation) * p0.vector()[:]

        if verbose and not converged:
            terminal_print(
                "    [{} IPCS] reached {} steps without meeting flow tolerances: du={:.3e}, dp={:.3e}".format(
                    label, step_limit, du_rel, dp_rel
                ),
                is_root=is_root,
            )

        return float(du_rel), float(dp_rel), converged

    root_print("Steady RANS-SA solve: pseudo-time IPCS flow + steady SA Picard coupling", is_root=is_root)
    if pressure_drop_description is not None:
        root_print("  Pressure-drop metric: {}.".format(pressure_drop_description), is_root=is_root)
    if convergence_type == "diagnostic_window_relative_change":
        root_print(
            "  Picard convergence metric: relative change in {} over {} Picard iterations <= {:.3e}.".format(
                convergence_diagnostic_key,
                convergence_window,
                convergence_relative_tolerance,
            ),
            is_root=is_root,
        )

    # Outer Picard loop: alternate between a flow solve at fixed nu_t and an
    # SA transport solve at fixed velocity until all coupled fields stop moving.
    converged = False
    for picard_iter in range(1, max_picard + 1):
        u_prev_picard.assign(u0)
        p_prev_picard.assign(p0)
        nu_tilde_prev_picard.assign(turbulence_model.nu_tilde0)

        if log_flow_iterations:
            terminal_print("  [Picard {}] flow solve".format(picard_iter), is_root=is_root)
        flow_du, flow_dp, flow_converged = solve_flow_to_steady(
            "Picard {}".format(picard_iter),
            verbose=log_flow_iterations,
        )

        if log_flow_iterations:
            terminal_print("  [Picard {}] steady SA solve".format(picard_iter), is_root=is_root)
        for sa_sweep in range(sa_sweeps):
            # Rebuild SA forms with the latest flow, solve nu_tilde, then relax it.
            turbulence_model.construct_forms(u0)
            turbulence_model.solve_turbulence_model()
            turbulence_model.update_variables(relaxation=sa_relaxation)
            if nu_tilde_floor is not None:
                bound_from_bellow(turbulence_model.nu_tilde0, float(nu_tilde_floor))
                turbulence_model.enforce_boundary_conditions()
                turbulence_model.nu_tilde1.assign(turbulence_model.nu_tilde0)

        picard_errors = [
            l2_norm_diff(u0, u_prev_picard, dx),
            l2_norm_diff(p0, p_prev_picard, dx),
            l2_norm_diff(turbulence_model.nu_tilde0, nu_tilde_prev_picard, dx),
        ]
        residuals["u"].append(float(picard_errors[0]))
        residuals["p"].append(float(picard_errors[1]))
        residuals["nu_tilde"].append(float(picard_errors[2]))
        residuals["flow_u"].append(float(flow_du))
        residuals["flow_p"].append(float(flow_dp))
        pressure_drop = None
        if pressure_drop_evaluator is not None:
            pressure_drop = pressure_drop_evaluator(p0)
            residuals["pressure_drop"].append(float(pressure_drop))
        diagnostic_values = []
        if diagnostic_specs:
            diagnostic_state = {
                "picard_iter": picard_iter,
                "u": u0,
                "p": p0,
                "nu_tilde": turbulence_model.nu_tilde0,
                "turbulence_model": turbulence_model,
            }
            for diagnostic in diagnostic_specs:
                diagnostic_value = float(diagnostic["evaluator"](diagnostic_state))
                residuals[diagnostic["key"]].append(diagnostic_value)
                diagnostic_values.append((diagnostic, diagnostic_value))

        diagnostic_window_relative_change = None
        if convergence_type == "diagnostic_window_relative_change":
            convergence_history = residuals[convergence_diagnostic_key]
            if len(convergence_history) > convergence_window:
                current_value = convergence_history[-1]
                previous_value = convergence_history[-1 - convergence_window]
                diagnostic_window_relative_change = (
                    abs(current_value - previous_value)
                    / max(abs(current_value), convergence_value_floor)
                )
        checkpoint_saved = False
        checkpoint_skipped = False
        if picard_checkpoint_every > 0 and picard_iter % picard_checkpoint_every == 0:
            if flow_converged or not checkpoint_require_flow_convergence:
                checkpoint_saved = save_picard_checkpoint()
            else:
                checkpoint_skipped = True

        if is_root:
            picard_message = (
                "Picard {:04d} ({:.2f}s): outer du={:.3e}, outer dp={:.3e}, "
                "outer dnu_tilde={:.3e}; "
                "flow du={:.3e}, flow dp={:.3e}".format(
                    picard_iter,
                    time.time() - start_time,
                    picard_errors[0],
                    picard_errors[1],
                    picard_errors[2],
                    flow_du,
                    flow_dp,
                )
            )
            if pressure_drop is not None:
                picard_message += "; {}={:.6e} {}".format(
                    pressure_drop_label,
                    pressure_drop,
                    pressure_drop_unit,
                )
            for diagnostic, diagnostic_value in diagnostic_values:
                picard_message += "; {}={}".format(
                    diagnostic["label"],
                    diagnostic["format"].format(diagnostic_value),
                )
                if diagnostic["unit"]:
                    picard_message += " {}".format(diagnostic["unit"])
            if convergence_type == "diagnostic_window_relative_change":
                if diagnostic_window_relative_change is None:
                    picard_message += "; {}_rel{}=pending".format(
                        convergence_diagnostic_key,
                        convergence_window,
                    )
                else:
                    picard_message += "; {}_rel{}={:.3e}".format(
                        convergence_diagnostic_key,
                        convergence_window,
                        diagnostic_window_relative_change,
                    )
            if not flow_converged:
                picard_message += "; flow solve did not meet tolerances before SA update"
            if checkpoint_saved:
                picard_message += "; checkpoint saved"
            if checkpoint_skipped:
                picard_message += "; checkpoint not saved because flow solve did not meet tolerances"
            print(picard_message)

        picard_converged = False
        if convergence_type == "field_norm":
            picard_converged = (
                flow_converged
                and picard_errors[0] <= tolerance_u
                and picard_errors[1] <= tolerance_p
                and picard_errors[2] <= tolerance_nu_tilde
            )
        elif convergence_type == "diagnostic_window_relative_change":
            picard_converged = (
                diagnostic_window_relative_change is not None
                and diagnostic_window_relative_change <= convergence_relative_tolerance
                and (flow_converged or not convergence_require_flow)
                and (picard_errors[0] <= tolerance_u or not convergence_require_velocity)
                and (picard_errors[1] <= tolerance_p or not convergence_require_pressure)
                and (picard_errors[2] <= tolerance_nu_tilde or not convergence_require_nu_tilde)
            )

        if picard_converged:
            converged = True
            convergence_message = (
                "Steady RANS-SA Picard solve converged in {} iterations ({:.2f}s); "
                "final flow residuals: du={:.3e}, dp={:.3e}".format(
                    picard_iter,
                    time.time() - start_time,
                    flow_du,
                    flow_dp,
                )
            )
            if pressure_drop is not None:
                convergence_message += "; {}={:.6e} {}".format(
                    pressure_drop_label,
                    pressure_drop,
                    pressure_drop_unit,
                )
            for diagnostic, diagnostic_value in diagnostic_values:
                convergence_message += "; {}={}".format(
                    diagnostic["label"],
                    diagnostic["format"].format(diagnostic_value),
                )
                if diagnostic["unit"]:
                    convergence_message += " {}".format(diagnostic["unit"])
            if diagnostic_window_relative_change is not None:
                convergence_message += "; {}_rel{}={:.3e}".format(
                    convergence_diagnostic_key,
                    convergence_window,
                    diagnostic_window_relative_change,
                )
            convergence_message += "."
            root_print(
                convergence_message,
                is_root=is_root,
            )
            break

    if not converged:
        root_print(
            "Warning: steady RANS-SA Picard solve reached {} iterations without full convergence.".format(
                max_picard
            ),
            is_root=is_root,
        )

    if log_flow_iterations:
        terminal_print("  [Final flow] IPCS with updated turbulent viscosity", is_root=is_root)
    final_flow_du, final_flow_dp, _ = solve_flow_to_steady(
        "Final flow",
        max_iters=final_flow_max_iters,
        verbose=log_flow_iterations,
    )
    residuals["flow_u"].append(float(final_flow_du))
    residuals["flow_p"].append(float(final_flow_dp))
    if pressure_drop_evaluator is not None:
        final_pressure_drop = pressure_drop_evaluator(p0)
        residuals["pressure_drop"].append(float(final_pressure_drop))
        root_print(
            "Final flow {}={:.6e} {}.".format(
                pressure_drop_label,
                final_pressure_drop,
                pressure_drop_unit,
            ),
            is_root=is_root,
        )

    solutions = {
        "u": u0,
        "p": p0,
        "nu_tilde": turbulence_model.nu_tilde0,
    }

    if post_processing.get("PLOT", False):
        visualize_functions(solutions)
        visualize_convergence(residuals)

    if post_processing.get("SAVE", False):
        for key, field in solutions.items():
            save_pvd_file(field, saving_directory["PVD_FILES"] + key + ".pvd")
            save_h5_checkpoint_field(field, saving_directory["H5_FILES"] + key + ".h5")

        for key, values in residuals.items():
            save_list(values, saving_directory["RESIDUALS"] + key + ".txt")

    return solutions, residuals
