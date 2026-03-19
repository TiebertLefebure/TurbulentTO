from dolfin import *
import argparse
import importlib.util
import numpy as np
import os
import sys
import time


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TURB_MODELS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "TurbulenceModels"))
if TURB_MODELS_DIR not in sys.path:
    sys.path.insert(0, TURB_MODELS_DIR)

from Utilities import (
    are_close_all,
    calculate_Distance_field,
    calculate_cfl_time_step,
    initialize_functions,
    save_h5_file,
    save_list,
    save_pvd_file,
    visualize_convergence,
    visualize_functions,
)
from TurbulenceModel_SpalartAllmaras import SpalartAllmarasTransient as SpalartAllmaras


DEFAULT_CONFIG_PATH = os.path.join(
    THIS_DIR, "Configs", "Config_PipeBendBorrvall_SpalartAllmaras.py"
)


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(THIS_DIR, path_value)


def _load_config_module(config_argument):
    if config_argument is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = config_argument
        if not os.path.isfile(config_path):
            candidate_in_this_dir = os.path.join(THIS_DIR, config_argument)
            candidate_in_configs = os.path.join(THIS_DIR, "Configs", config_argument)
            if os.path.isfile(candidate_in_this_dir):
                config_path = candidate_in_this_dir
            else:
                config_path = candidate_in_configs

    if not os.path.isfile(config_path):
        raise FileNotFoundError("Could not find config file: {}".format(config_argument))

    module_name = os.path.splitext(os.path.basename(config_path))[0].replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load config module from '{}'".format(config_path))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, config_path


def _build_parser():
    parser = argparse.ArgumentParser(description="Standalone SA simulation for Borrvall pipe-bend domain.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (defaults to Configs/Config_PipeBendBorrvall_SpalartAllmaras.py).",
    )
    parser.add_argument("--nx", type=int, default=None, help="Override mesh cells in x-direction.")
    parser.add_argument("--ny", type=int, default=None, help="Override mesh cells in y-direction.")
    parser.add_argument("--dt", type=float, default=None, help="Override initial time step.")
    parser.add_argument("--max-iters", type=int, default=None, help="Override MAX_ITERATIONS.")
    return parser


def _apply_overrides(config_module, args):
    if args.nx is not None:
        setattr(config_module, "NX", int(args.nx))
    if args.ny is not None:
        setattr(config_module, "NY", int(args.ny))
    if args.dt is not None:
        config_module.simulation_prm["STEP_SIZE"] = float(args.dt)
    if args.max_iters is not None:
        config_module.simulation_prm["MAX_ITERATIONS"] = int(args.max_iters)



### --------------------------------------------------------------------------------------
### Diagnostic tests for solution performance assessment

def _print_end_diagnostics(config, mesh, ds, velocity, nu_tilde, residuals):
    tol = config.simulation_prm["TOLERANCE"]
    print("---- End-of-run diagnostics ----")
    for key, values in residuals.items():
        if len(values) == 0:
            print("residual {}: no entries".format(key))
            continue
        final_value = float(values[-1])
        print(
            "residual {}: final={:.3e}, iters={}, pass={}".format(
                key, final_value, len(values), final_value <= tol
            )
        )

    n = FacetNormal(mesh)
    inlet_flux = 0.0
    outlet_flux = 0.0
    for marker in _as_list(config.boundary_markers.get("INFLOW", [])):
        inlet_flux += -assemble(dot(velocity, n) * ds(marker))
    for marker in _as_list(config.boundary_markers.get("OUTFLOW", [])):
        outlet_flux += assemble(dot(velocity, n) * ds(marker))

    rel_imbalance = abs(inlet_flux - outlet_flux) / max(
        abs(inlet_flux), abs(outlet_flux), 1.0e-16
    )
    net_boundary_flux = assemble(dot(velocity, n) * ds)
    print(
        "flux: q_in={:.6e}, q_out={:.6e}, rel_imbalance={:.3e}, net_boundary_flux={:.3e}".format(
            inlet_flux, outlet_flux, rel_imbalance, net_boundary_flux
        )
    )

    if hasattr(config, "U_MAX_INLET") and hasattr(config, "INLET_WIDTH"):
        target_flux = 2.0 * float(config.INLET_WIDTH) * float(config.U_MAX_INLET) / 3.0
        print("flux target (single parabolic inlet): {:.6e}".format(target_flux))

    nu_values = nu_tilde.vector().get_local()
    if nu_values.size > 0:
        print(
            "nu_tilde stats: min={:.3e}, p99={:.3e}, max={:.3e}".format(
                float(np.min(nu_values)),
                float(np.percentile(nu_values, 99.0)),
                float(np.max(nu_values)),
            )
        )

### --------------------------------------------------------------------------------------

def main():
    args = _build_parser().parse_args()
    config, config_path = _load_config_module(args.config)
    _apply_overrides(config, args)
    print("Using config: {}".format(config_path))

    if hasattr(config, "create_mesh_and_boundaries"):
        mesh, marked_facets = config.create_mesh_and_boundaries()
    else:
        mesh = config.create_mesh()
        marked_facets = config.mark_boundaries(mesh)

    quadrature_degree = config.simulation_prm["QUADRATURE_DEGREE"]
    dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
    ds = Measure(
        "ds",
        domain=mesh,
        subdomain_data=marked_facets,
        metadata={"quadrature_degree": quadrature_degree},
    )

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    N = FunctionSpace(mesh, "CG", 1)

    # Build all Dirichlet BCs from config dictionaries.
    bcu = []
    bcp = []
    bcn = []
    bc_specs = [("U", bcu, V), ("P", bcp, Q), ("NU_TILDE", bcn, N)]
    for boundary_name, markers in config.boundary_markers.items():
        if markers is None:
            continue
        for marker in _as_list(markers):
            for variable, bc_list, space in bc_specs:
                value = config.boundary_conditions[boundary_name].get(variable)
                if value is not None:
                    bc_list.append(DirichletBC(space, value, marked_facets, marker))

    if getattr(config, "ENABLE_PRESSURE_PIN", True):
        pin_x, pin_y = getattr(config, "PRESSURE_PIN_POINT", (0.0, 0.0))
        pin_expr = "near(x[0], {:.16g}) && near(x[1], {:.16g})".format(float(pin_x), float(pin_y))
        bcp.append(DirichletBC(Q, Constant(0.0), pin_expr, "pointwise"))

    nu = Constant(config.physical_prm["VISCOSITY"])
    force = Constant(config.physical_prm["FORCE"])
    dt = Constant(config.simulation_prm["STEP_SIZE"])

    y = calculate_Distance_field(
        N,
        marked_facets,
        _as_list(config.boundary_markers["WALLS"]),
        config.simulation_prm.get("DISTANCE_RELAXATION", 0.01),
    )

    u, v, u1, u0 = initialize_functions(V, Constant(config.initial_conditions["U"]))
    p, q, p1, p0 = initialize_functions(Q, Constant(config.initial_conditions["P"]))

    turbulence_model = SpalartAllmaras(
        N,
        bcn,
        config.initial_conditions["NU_TILDE"],
        nu,
        force,
        dx,
        ds,
        dt,
        y,
    )
    turbulence_model.construct_forms(u1)

    h = CellDiameter(mesh)
    u_mag = sqrt(dot(u0, u0) + 1.0e-10)
    tau = h / (2.0 * u_mag)
    residual = (u - u0) / dt + dot(u0, nabla_grad(u)) - force
    F_supg = inner(tau * dot(u0, nabla_grad(v)), residual) * dx

    F1 = (
        dot((u - u0) / dt, v) * dx
        + dot(dot(u0, nabla_grad(u)), v) * dx
        + inner((nu + turbulence_model.nu_t) * grad(u), grad(v)) * dx
        - dot(force, v) * dx
        + F_supg
    )
    F2 = dot(grad(p), grad(q)) * dx + dot(div(u1) / dt, q) * dx
    F3 = dot(u, v) * dx - dot(u1, v) * dx + dt * dot(grad(p1), v) * dx

    a_1, l_1 = lhs(F1), rhs(F1)
    a_2, l_2 = lhs(F2), rhs(F2)
    a_3, l_3 = lhs(F3), rhs(F3)

    u_relax = config.simulation_prm.get("U_RELAXATION_FACTOR", 1.0)
    nut_relax = config.simulation_prm.get("NUT_RELAXATION_FACTOR", 1.0)
    linear_solver = config.simulation_prm.get("LINEAR_SOLVER", "mumps")

    residuals = {key: [] for key in ["u", "p", "nu_tilde"]}
    start_time = time.time()
    converged = False

    for iteration in range(config.simulation_prm["MAX_ITERATIONS"]):
        print("--- iter {:03d} ---".format(iteration + 1))
        if iteration > 0:
            step_size = calculate_cfl_time_step(
                u0,
                MaxCellEdgeLength(mesh),
                MinCellEdgeLength(mesh),
                config.simulation_prm["CFL_RELAXATION"],
                mesh,
            )
            min_step_size = config.simulation_prm.get("MIN_STEP_SIZE", None)
            max_step_size = config.simulation_prm.get("MAX_STEP_SIZE", None)
            if min_step_size is not None:
                step_size = max(step_size, min_step_size)
            if max_step_size is not None:
                step_size = min(step_size, max_step_size)
            dt.assign(Constant(step_size))

        print("  [Forward solve]")
        A_1 = assemble(a_1)
        b_1 = assemble(l_1)
        for bc in bcu:
            bc.apply(A_1, b_1)
        solve(A_1, u1.vector(), b_1, linear_solver)

        A_2 = assemble(a_2)
        b_2 = assemble(l_2)
        for bc in bcp:
            bc.apply(A_2, b_2)
        solve(A_2, p1.vector(), b_2, linear_solver)

        A_3 = assemble(a_3)
        b_3 = assemble(l_3)
        for bc in bcu:
            bc.apply(A_3, b_3)
        solve(A_3, u1.vector(), b_3, linear_solver)

        print("  [SA solve]")
        turbulence_model.solve_turbulence_model()

        break_flag, errors = are_close_all(
            [u1, p1, turbulence_model.nu_tilde1],
            [u0, p0, turbulence_model.nu_tilde0],
            config.simulation_prm["TOLERANCE"],
        )
        print(
            "iter: {} ({:.2f}s, dt = {:.2e}s) --- L2 errors: |u1-u0|= {:.2e}, |p1-p0|= {:.2e}, "
            "|nu_tilde1-nu_tilde0|= {:.2e} (required: {:.2e})".format(
                iteration + 1,
                time.time() - start_time,
                float(dt),
                errors[0],
                errors[1],
                errors[2],
                config.simulation_prm["TOLERANCE"],
            )
        )

        for key, error in zip(residuals.keys(), errors):
            residuals[key].append(error)

        u0.assign((1.0 - u_relax) * u0 + u_relax * u1)
        p0.assign(p1)
        turbulence_model.update_variables(relaxation=nut_relax)

        if break_flag:
            converged = True
            print(
                "Simulation converged in {} iterations ({:.2f} seconds)".format(
                    iteration + 1, time.time() - start_time
                )
            )
            break

    if not converged:
        print(
            "Reached MAX_ITERATIONS={} without satisfying tolerance {:.2e}.".format(
                config.simulation_prm["MAX_ITERATIONS"], config.simulation_prm["TOLERANCE"]
            )
        )

    solutions = {"u": u1, "p": p1, "nu_tilde": turbulence_model.nu_tilde1}
    _print_end_diagnostics(config, mesh, ds, u1, turbulence_model.nu_tilde1, residuals)

    if config.post_processing.get("SAVE", False):
        for key, function_value in solutions.items():
            save_pvd_file(function_value, _resolve_path(config.saving_directory["PVD_FILES"] + key + ".pvd"))
            save_h5_file(function_value, _resolve_path(config.saving_directory["H5_FILES"] + key + ".h5"))
        for key, values in residuals.items():
            save_list(values, _resolve_path(config.saving_directory["RESIDUALS"] + key + ".txt"))

    if config.post_processing.get("PLOT", False):
        visualize_functions(solutions)
        visualize_convergence(residuals)


if __name__ == "__main__":
    main()
