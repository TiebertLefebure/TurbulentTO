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

    residuals = {key: [] for key in ["u", "p", "nu_tilde", "flow_u", "flow_p"]}
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

    def solve_flow_to_steady(label, max_iters=None):
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

            if is_root and (flow_iter == 1 or flow_iter % log_every == 0):
                print(
                    "    [{} IPCS {:04d}] du={:.3e}, dp={:.3e}".format(
                        label, flow_iter, du_rel, dp_rel
                    )
                )

            if du_rel <= flow_tolerance_u and dp_rel <= flow_tolerance_p:
                u0.assign(u1)
                p0.assign(p1)
                converged = True
                break

            # Under-relax the flow fields before the next pseudo-time IPCS step.
            u0.vector()[:] = u_relaxation * u1.vector()[:] + (1.0 - u_relaxation) * u0.vector()[:]
            p0.vector()[:] = p_relaxation * p1.vector()[:] + (1.0 - p_relaxation) * p0.vector()[:]

        if not converged:
            root_print(
                "    [{} IPCS] reached {} steps without meeting flow tolerances: du={:.3e}, dp={:.3e}".format(
                    label, step_limit, du_rel, dp_rel
                ),
                is_root=is_root,
            )

        return float(du_rel), float(dp_rel), converged

    root_print("Steady RANS-SA solve: pseudo-time IPCS flow + steady SA Picard coupling", is_root=is_root)
    root_print("  SA equation: steady Galerkin, no SA SUPG term", is_root=is_root)

    # Outer Picard loop: alternate between a flow solve at fixed nu_t and an
    # SA transport solve at fixed velocity until all coupled fields stop moving.
    converged = False
    for picard_iter in range(1, max_picard + 1):
        u_prev_picard.assign(u0)
        p_prev_picard.assign(p0)
        nu_tilde_prev_picard.assign(turbulence_model.nu_tilde0)

        root_print("  [Picard {}] flow solve".format(picard_iter), is_root=is_root)
        flow_du, flow_dp, flow_converged = solve_flow_to_steady("Picard {}".format(picard_iter))

        root_print("  [Picard {}] steady SA solve".format(picard_iter), is_root=is_root)
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

        if is_root:
            print(
                "  Picard {:04d} ({:.2f}s): outer du={:.3e}, outer dp={:.3e}, "
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

        if (
            picard_errors[0] <= tolerance_u
            and picard_errors[1] <= tolerance_p
            and picard_errors[2] <= tolerance_nu_tilde
        ):
            converged = True
            root_print(
                "Steady RANS-SA Picard solve converged in {} iterations ({:.2f}s); "
                "final flow residuals: du={:.3e}, dp={:.3e}.".format(
                    picard_iter,
                    time.time() - start_time,
                    flow_du,
                    flow_dp,
                ),
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

    root_print("  [Final flow] IPCS with updated turbulent viscosity", is_root=is_root)
    final_flow_du, final_flow_dp, _ = solve_flow_to_steady(
        "Final flow",
        max_iters=final_flow_max_iters,
    )
    residuals["flow_u"].append(float(final_flow_du))
    residuals["flow_p"].append(float(final_flow_dp))

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
            save_h5_file(field, saving_directory["H5_FILES"] + key + ".h5")

        for key, values in residuals.items():
            save_list(values, saving_directory["RESIDUALS"] + key + ".txt")

    return solutions, residuals
