from dolfin import *
from Utilities import *
from Configs.ConfigBackStep_SpalartAllmaras_Steady import *
from TurbulenceModel_SpalartAllmaras import SpalartAllmarasSteadyState as SpalartAllmaras
from SA_Steady_IPCS_Picard_Solver import run_steady_sa_ipcs_picard

import os
import numpy as np


parameters["std_out_all_processes"] = False
IS_ROOT = MPI.COMM_WORLD.Get_rank() == 0


def _as_marker_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _boundary_average(field, markers, ds_measure):
    weighted_value = 0.0
    boundary_measure = 0.0
    for marker in _as_marker_list(markers):
        marker_id = int(marker)
        marker_measure = float(assemble(Constant(1.0) * ds_measure(marker_id)))
        weighted_value += float(assemble(field * ds_measure(marker_id)))
        boundary_measure += marker_measure
    if abs(boundary_measure) <= 1.0e-30:
        return float("nan")
    return weighted_value / boundary_measure


def _boundary_flux(velocity, markers, ds_measure):
    normal = FacetNormal(velocity.function_space().mesh())
    flux = 0.0
    for marker in _as_marker_list(markers):
        flux += float(assemble(dot(velocity, normal) * ds_measure(int(marker))))
    return flux


def _relative_mass_imbalance(velocity, inlet_markers, outlet_markers, ds_measure):
    inlet_flux = _boundary_flux(velocity, inlet_markers, ds_measure)
    outlet_flux = _boundary_flux(velocity, outlet_markers, ds_measure)
    flux_scale = max(abs(inlet_flux), abs(outlet_flux), 1.0e-30)
    return abs(inlet_flux + outlet_flux) / flux_scale


def _first_negative_to_positive_crossing(xs, values):
    xs = np.asarray(xs, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(values)
    xs = xs[valid]
    values = values[valid]
    if xs.size < 2:
        return None

    for i in range(xs.size - 1):
        left = values[i]
        right = values[i + 1]
        if left <= 0.0 and right > 0.0:
            if abs(right - left) <= 1.0e-30:
                return float(xs[i])
            fraction = -left / (right - left)
            return float(xs[i] + fraction * (xs[i + 1] - xs[i]))
    return None


def _negative_interval_length(xs, values):
    xs = np.asarray(xs, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(values)
    xs = xs[valid]
    values = values[valid]
    if xs.size < 2:
        return float("nan")

    length = 0.0
    for i in range(xs.size - 1):
        x0 = xs[i]
        x1 = xs[i + 1]
        y0 = values[i]
        y1 = values[i + 1]
        dx_segment = x1 - x0

        if y0 < 0.0 and y1 < 0.0:
            length += dx_segment
        elif y0 < 0.0 <= y1 and abs(y1 - y0) > 1.0e-30:
            length += dx_segment * (-y0 / (y1 - y0))
        elif y0 >= 0.0 > y1 and abs(y1 - y0) > 1.0e-30:
            length += dx_segment * (1.0 - (-y0 / (y1 - y0)))
    return float(length)


def _project_scalar(expr, space, name):
    field = project(expr, space)
    field.rename(name, name)
    return field


def _sample_lower_wall(velocity, scalar_space, metrics_config, viscosity):
    step_x = float(metrics_config["STEP_X"])
    outlet_x = float(metrics_config["OUTLET_X"])
    lower_wall_y = float(metrics_config["LOWER_WALL_Y"])
    wall_y_offset = float(metrics_config["WALL_SAMPLE_Y_OFFSET"])
    wall_x_offset = float(metrics_config["WALL_SAMPLE_X_OFFSET"])
    sample_count = max(2, int(metrics_config["WALL_SAMPLE_COUNT"]))
    u_reference = float(metrics_config["U_REFERENCE"])

    dudy = _project_scalar(Dx(velocity[0], 1), scalar_space, "du_dy_lower_wall")
    xs = np.linspace(step_x + wall_x_offset, outlet_x - wall_x_offset, sample_count)
    y_sample = lower_wall_y + wall_y_offset

    shear_gradient_samples = []
    skin_friction_samples = []
    near_wall_u_samples = []
    for x_value in xs:
        point = Point(float(x_value), y_sample)
        try:
            shear_gradient = float(dudy(point))
            velocity_value = velocity(point)
            near_wall_u = float(velocity_value[0])
        except Exception:
            shear_gradient = float("nan")
            near_wall_u = float("nan")

        if abs(u_reference) > 1.0e-30 and np.isfinite(shear_gradient):
            skin_friction = 2.0 * viscosity * shear_gradient / (u_reference ** 2)
        else:
            skin_friction = float("nan")

        shear_gradient_samples.append(shear_gradient)
        skin_friction_samples.append(skin_friction)
        near_wall_u_samples.append(near_wall_u)

    return {
        "x": xs,
        "y": y_sample,
        "du_dy": np.asarray(shear_gradient_samples, dtype=float),
        "cf": np.asarray(skin_friction_samples, dtype=float),
        "near_wall_u": np.asarray(near_wall_u_samples, dtype=float),
    }


def _reattachment_length_over_h_for_velocity(velocity, scalar_space, metrics_config, viscosity):
    wall_samples = _sample_lower_wall(
        velocity,
        scalar_space,
        metrics_config,
        viscosity,
    )
    reattachment_x = _first_negative_to_positive_crossing(
        wall_samples["x"],
        wall_samples["du_dy"],
    )
    if reattachment_x is None:
        return float("nan")
    return (
        reattachment_x - float(metrics_config["STEP_X"])
    ) / float(metrics_config["STEP_HEIGHT"])


def _finite_min(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.min(finite)) if finite.size else float("nan")


def _finite_max(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else float("nan")


def _format_metric(value, unit=""):
    if value is None:
        return "not found"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return "not found"
    formatted = "{:.6e}".format(value_float)
    return "{} {}".format(formatted, unit).strip()


def _build_backstep_picard_diagnostics(
    *,
    scalar_space,
    metrics_config,
    pressure_metric,
    ds_measure,
    viscosity,
):
    def reattachment_length_over_h(state):
        return _reattachment_length_over_h_for_velocity(
            state["u"],
            scalar_space,
            metrics_config,
            viscosity,
        )

    def mass_imbalance(state):
        return _relative_mass_imbalance(
            state["u"],
            pressure_metric.get("INLET_MARKERS"),
            pressure_metric.get("OUTLET_MARKERS"),
            ds_measure,
        )

    return [
        {
            "key": "reattachment_length_over_h",
            "label": "x_r/h",
            "format": "{:.6e}",
            "evaluator": reattachment_length_over_h,
        },
        {
            "key": "mass_imbalance",
            "label": "mass imbalance",
            "format": "{:.3e}",
            "evaluator": mass_imbalance,
        },
    ]


def _compute_backstep_metrics(
    *,
    solutions,
    turbulence_model,
    scalar_space,
    metric_space,
    dx_measure,
    ds_measure,
    metrics_config,
    pressure_metric,
    viscosity,
    saving_directory,
):
    velocity = solutions["u"]
    pressure = solutions["p"]
    density = float(metrics_config["DENSITY"])
    step_x = float(metrics_config["STEP_X"])
    step_height = float(metrics_config["STEP_HEIGHT"])

    x = SpatialCoordinate(velocity.function_space().mesh())
    downstream_indicator = conditional(gt(x[0], Constant(step_x)), Constant(1.0), Constant(0.0))
    reverse_indicator = conditional(
        gt(x[0], Constant(step_x)),
        conditional(lt(velocity[0], Constant(0.0)), Constant(1.0), Constant(0.0)),
        Constant(0.0),
    )

    strain = sym(grad(velocity))
    nu_t = turbulence_model.nu_t
    total_dissipation_density = (
        Constant(2.0 * density)
        * (Constant(viscosity) + nu_t)
        * inner(strain, strain)
    )
    turbulent_dissipation_density = (
        Constant(2.0 * density)
        * nu_t
        * inner(strain, strain)
    )

    domain_area = float(assemble(Constant(1.0) * dx_measure))
    downstream_area = float(assemble(downstream_indicator * dx_measure))
    reverse_flow_area = float(assemble(reverse_indicator * dx_measure))
    total_dissipation = float(assemble(total_dissipation_density * dx_measure))
    turbulent_dissipation = float(assemble(turbulent_dissipation_density * dx_measure))
    reverse_flow_dissipation = float(assemble(total_dissipation_density * reverse_indicator * dx_measure))

    inlet_pressure = _boundary_average(
        pressure,
        pressure_metric.get("INLET_MARKERS"),
        ds_measure,
    )
    outlet_pressure = _boundary_average(
        pressure,
        pressure_metric.get("OUTLET_MARKERS"),
        ds_measure,
    )
    pressure_scale = float(pressure_metric.get("PRESSURE_SCALE", 1.0))
    pressure_drop = pressure_scale * (inlet_pressure - outlet_pressure)
    mass_imbalance = _relative_mass_imbalance(
        velocity,
        pressure_metric.get("INLET_MARKERS"),
        pressure_metric.get("OUTLET_MARKERS"),
        ds_measure,
    )

    nu_t_field = _project_scalar(nu_t, scalar_space, "nu_t")
    nu_t_local = nu_t_field.vector().get_local()
    max_nu_t = _finite_max(nu_t_local)
    mean_nu_t = float(assemble(nu_t * dx_measure)) / max(domain_area, 1.0e-30)

    wall_samples = _sample_lower_wall(
        velocity,
        scalar_space,
        metrics_config,
        viscosity,
    )
    reattachment_x = _first_negative_to_positive_crossing(
        wall_samples["x"],
        wall_samples["du_dy"],
    )
    reverse_flow_length = _negative_interval_length(
        wall_samples["x"],
        wall_samples["near_wall_u"],
    )
    wall_shear_negative_length = _negative_interval_length(
        wall_samples["x"],
        wall_samples["du_dy"],
    )

    if reattachment_x is None:
        reattachment_length_over_h = float("nan")
    else:
        reattachment_length_over_h = (reattachment_x - step_x) / step_height

    metric_fields = {
        "nu_t": nu_t_field,
        "reverse_flow_indicator": _project_scalar(
            reverse_indicator,
            metric_space,
            "reverse_flow_indicator",
        ),
        "dissipation_density": _project_scalar(
            total_dissipation_density,
            metric_space,
            "dissipation_density",
        ),
        "turbulent_dissipation_density": _project_scalar(
            turbulent_dissipation_density,
            metric_space,
            "turbulent_dissipation_density",
        ),
    }

    if metrics_config.get("SAVE_METRIC_FIELDS", False):
        for key, field in metric_fields.items():
            save_pvd_file(field, saving_directory["PVD_FILES"] + key + ".pvd")

    return {
        "domain_area": domain_area,
        "downstream_area": downstream_area,
        "reverse_flow_area": reverse_flow_area,
        "reverse_flow_area_over_h2": reverse_flow_area / (step_height ** 2),
        "reverse_flow_area_fraction_downstream": reverse_flow_area / max(downstream_area, 1.0e-30),
        "pressure_drop": pressure_drop,
        "pressure_unit": pressure_metric.get("UNIT", ""),
        "mass_imbalance": mass_imbalance,
        "inlet_pressure_average": inlet_pressure,
        "outlet_pressure_average": outlet_pressure,
        "reattachment_x": reattachment_x,
        "reattachment_length_over_h": reattachment_length_over_h,
        "near_wall_reverse_flow_length": reverse_flow_length,
        "near_wall_reverse_flow_length_over_h": reverse_flow_length / step_height,
        "wall_shear_negative_length": wall_shear_negative_length,
        "wall_shear_negative_length_over_h": wall_shear_negative_length / step_height,
        "min_lower_wall_du_dy": _finite_min(wall_samples["du_dy"]),
        "max_lower_wall_du_dy": _finite_max(wall_samples["du_dy"]),
        "min_lower_wall_cf": _finite_min(wall_samples["cf"]),
        "max_lower_wall_cf": _finite_max(wall_samples["cf"]),
        "mean_nu_t": mean_nu_t,
        "max_nu_t": max_nu_t,
        "mean_nu_t_over_nu": mean_nu_t / viscosity,
        "max_nu_t_over_nu": max_nu_t / viscosity,
        "total_dissipation_proxy": total_dissipation,
        "turbulent_dissipation_proxy": turbulent_dissipation,
        "reverse_flow_dissipation_proxy": reverse_flow_dissipation,
        "reverse_flow_dissipation_fraction": reverse_flow_dissipation / max(total_dissipation, 1.0e-30),
        "wall_sample_y": wall_samples["y"],
        "wall_sample_count": len(wall_samples["x"]),
        "wall_samples": wall_samples,
    }


def _write_backstep_metrics(metrics, metrics_config, saving_directory):
    if not IS_ROOT:
        return

    results_root = simulation_results_root(saving_directory)
    os.makedirs(results_root, exist_ok=True)
    metrics_path = os.path.join(results_root, "BackStepMetrics.txt")
    sample_path = os.path.join(results_root, "BackStepWallSamples.csv")

    lines = [
        "",
        "Backward-facing-step separation metrics",
        "  Reattachment x_r: {}".format(_format_metric(metrics["reattachment_x"])),
        "  Reattachment length x_r/h: {}".format(_format_metric(metrics["reattachment_length_over_h"])),
        "  Near-wall reverse-flow length: {}".format(
            _format_metric(metrics["near_wall_reverse_flow_length"])
        ),
        "  Near-wall reverse-flow length/h: {}".format(
            _format_metric(metrics["near_wall_reverse_flow_length_over_h"])
        ),
        "  Negative wall-shear length/h: {}".format(
            _format_metric(metrics["wall_shear_negative_length_over_h"])
        ),
        "  Reverse-flow area: {}".format(_format_metric(metrics["reverse_flow_area"])),
        "  Reverse-flow area/h^2: {}".format(_format_metric(metrics["reverse_flow_area_over_h2"])),
        "  Reverse-flow downstream area fraction: {}".format(
            _format_metric(metrics["reverse_flow_area_fraction_downstream"])
        ),
        "  Static pressure drop: {}".format(
            _format_metric(metrics["pressure_drop"], metrics["pressure_unit"])
        ),
        "  Mass imbalance: {}".format(_format_metric(metrics["mass_imbalance"])),
        "  Mean nu_t/nu: {}".format(_format_metric(metrics["mean_nu_t_over_nu"])),
        "  Max nu_t/nu: {}".format(_format_metric(metrics["max_nu_t_over_nu"])),
        "  Total strain-rate dissipation proxy: {}".format(
            _format_metric(metrics["total_dissipation_proxy"])
        ),
        "  Turbulent-viscosity dissipation proxy: {}".format(
            _format_metric(metrics["turbulent_dissipation_proxy"])
        ),
        "  Reverse-flow dissipation fraction: {}".format(
            _format_metric(metrics["reverse_flow_dissipation_fraction"])
        ),
        "  Lower-wall sample y: {}".format(_format_metric(metrics["wall_sample_y"])),
        "  Lower-wall sample count: {}".format(metrics["wall_sample_count"]),
    ]

    print("\n".join(lines))
    with open(metrics_path, "w") as output:
        output.write("\n".join(lines).lstrip() + "\n")

    if metrics_config.get("SAVE_WALL_SAMPLE_TABLE", False):
        samples = metrics["wall_samples"]
        with open(sample_path, "w") as output:
            output.write("x,y,du_dy,cf,near_wall_u\n")
            for x_value, du_dy, cf_value, near_wall_u in zip(
                samples["x"],
                samples["du_dy"],
                samples["cf"],
                samples["near_wall_u"],
            ):
                output.write(
                    "{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}\n".format(
                        float(x_value),
                        float(samples["y"]),
                        float(du_dy),
                        float(cf_value),
                        float(near_wall_u),
                    )
                )

    print("Backstep metrics written to: {}".format(os.path.abspath(metrics_path)))
    if metrics_config.get("SAVE_WALL_SAMPLE_TABLE", False):
        print("Backstep wall samples written to: {}".format(os.path.abspath(sample_path)))


# Use steady SA-specific parameters from the config file.
simulation_prm = steady_sa_solver_parameters
setup_simulation_log(saving_directory, __file__)

# Load mesh.
mesh, marked_facets = load_mesh_from_file(mesh_files["MESH_DIRECTORY"], mesh_files["FACET_DIRECTORY"])

# Custom integration measures.
quadrature_degree = simulation_prm["QUADRATURE_DEGREE"]
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
ds = Measure(
    "ds",
    domain=mesh,
    subdomain_data=marked_facets,
    metadata={"quadrature_degree": quadrature_degree},
)

# Construct function spaces.
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
K = FunctionSpace(mesh, "CG", 1)
DG0 = FunctionSpace(mesh, "DG", 0)

# Construct boundary conditions.
bcu = []
bcp = []
bcn = []

for boundary_name, markers in boundary_markers.items():
    if markers is None:
        continue

    for marker in markers:
        for variable, bc_list, function_space in zip(
            ["U", "P", "NU_TILDE"],
            [bcu, bcp, bcn],
            [V, Q, K],
        ):
            condition_value = boundary_conditions[boundary_name].get(variable)
            if condition_value is None:
                continue

            if boundary_name == "SYMMETRY" and variable == "U":
                bc_list.append(DirichletBC(function_space.sub(1), condition_value, marked_facets, marker))
            else:
                bc_list.append(DirichletBC(function_space, condition_value, marked_facets, marker))

# Initialize constants and wall distance.
nu = Constant(physical_prm["VISCOSITY"])
force = Constant(physical_prm["FORCE"])
dt = Constant(float(simulation_prm["FLOW_IPCS_TIME_STEP"]))
y = calculate_Distance_field(
    K,
    marked_facets,
    boundary_markers["WALLS"],
    relax=WALL_DISTANCE_EIKONAL_RELAXATION,
    method=WALL_DISTANCE_METHOD,
    sigma_w=WALL_DISTANCE_YOON_SIGMA_W,
    g0=WALL_DISTANCE_YOON_G0,
    g_floor=WALL_DISTANCE_YOON_G_FLOOR,
    newton_rtol=WALL_DISTANCE_YOON_NEWTON_RTOL,
    newton_atol=WALL_DISTANCE_YOON_NEWTON_ATOL,
    newton_max_iters=WALL_DISTANCE_YOON_NEWTON_MAX_ITERATIONS,
    newton_relax=WALL_DISTANCE_YOON_NEWTON_RELAXATION,
    custom_dx=dx,
)

# Initialize functions.
u, v, u1, u0 = initialize_functions(V, Constant(initial_conditions["U"]))
p, q, p1, p0 = initialize_functions(Q, Constant(initial_conditions["P"]))

# Initialize steady SA model.
sa_options = {
    "LINEAR_SOLVER": simulation_prm.get("SA_TRANSPORT_LINEAR_SOLVER", "default"),
    "LINEAR_PRECONDITIONER": simulation_prm.get("SA_TRANSPORT_LINEAR_PRECONDITIONER", "default"),
}
turbulence_model = SpalartAllmaras(
    K,
    bcn,
    initial_conditions["NU_TILDE"],
    nu,
    force,
    dx,
    ds,
    y,
    sa_options=sa_options,
)
turbulence_model.construct_forms(u0)

# Pseudo-time incremental IPCS flow forms, without momentum SUPG.
F1 = (
    dot((u - u0) / dt, v) * dx
    + dot(dot(u0, nabla_grad(u)), v) * dx
    + inner((nu + turbulence_model.nu_t) * grad(u), grad(v)) * dx
    + dot(grad(p0), v) * dx
    - dot(force, v) * dx
)
F2 = dot(grad(p - p0), grad(q)) * dx + dot(div(u1) / dt, q) * dx
F3 = dot(u, v) * dx - dot(u1, v) * dx + dt * dot(grad(p1 - p0), v) * dx

a_1, l_1 = lhs(F1), rhs(F1)
a_2, l_2 = lhs(F2), rhs(F2)
a_3, l_3 = lhs(F3), rhs(F3)

if IS_ROOT:
    print("BackStep steady SA setup")
    print("  Mesh: {}".format(mesh_files["MESH_DIRECTORY"]))
    print("  Re_h: {:.6e}".format(REYNOLDS_NUMBER_STEP))
    print("  Re_Hin: {:.6e}".format(REYNOLDS_NUMBER_INLET_HEIGHT))
    print("  Target first-layer height for y+ ~= 1: {:.6e} m".format(TARGET_FIRST_LAYER_HEIGHT))
    print("  Inlet nu_tilde: {:.6e}".format(INLET_NU_TILDE))
    print("  Inlet eddy-viscosity ratio estimate: {:.6e}".format(INLET_EDDY_VISCOSITY_RATIO))
    print("  Wall-distance method: {}".format(WALL_DISTANCE_METHOD))
    print("  Symmetry/slip boundary: normal velocity fixed, nu_tilde natural")

solutions, residuals = run_steady_sa_ipcs_picard(
    simulation_prm=simulation_prm,
    post_processing=post_processing,
    saving_directory=saving_directory,
    dx=dx,
    dt=dt,
    a_1=a_1,
    l_1=l_1,
    a_2=a_2,
    l_2=l_2,
    a_3=a_3,
    l_3=l_3,
    bcu=bcu,
    bcp=bcp,
    u0=u0,
    u1=u1,
    p0=p0,
    p1=p1,
    velocity_space=V,
    pressure_space=Q,
    turbulence_space=K,
    turbulence_model=turbulence_model,
    normalize_pressure_mean=bool(simulation_prm.get("FLOW_IPCS_NORMALIZE_PRESSURE_MEAN", False)),
    ds=ds,
    pressure_drop_metric=pressure_drop_metric,
    picard_diagnostics=_build_backstep_picard_diagnostics(
        scalar_space=K,
        metrics_config=BACKSTEP_METRICS,
        pressure_metric=pressure_drop_metric,
        ds_measure=ds,
        viscosity=float(physical_prm["VISCOSITY"]),
    ),
    is_root=IS_ROOT,
)

# Rebuild SA terms against the final velocity before evaluating eddy-viscosity
# and dissipation diagnostics.
turbulence_model.construct_forms(solutions["u"])
backstep_metrics = _compute_backstep_metrics(
    solutions=solutions,
    turbulence_model=turbulence_model,
    scalar_space=K,
    metric_space=DG0,
    dx_measure=dx,
    ds_measure=ds,
    metrics_config=BACKSTEP_METRICS,
    pressure_metric=pressure_drop_metric,
    viscosity=float(physical_prm["VISCOSITY"]),
    saving_directory=saving_directory,
)
_write_backstep_metrics(backstep_metrics, BACKSTEP_METRICS, saving_directory)
