import math
import os

import numpy as np
from dolfin import Function, MPI, Point, TestFunction, assemble, dot, dx, project, sqrt


def _is_root(comm):
    return MPI.rank(comm) == 0


def _first_config_value(config, names, default=None):
    for name in names:
        if name in config:
            return config[name]
    return default


def _domain_bounds(config, density_space):
    coords = density_space.tabulate_dof_coordinates().reshape((-1, density_space.mesh().geometry().dim()))
    x_min = _first_config_value(config, ("DILGEN_PAPER_X_MIN", "DOMAIN_X_MIN"), float(np.min(coords[:, 0])))
    x_max = _first_config_value(config, ("DILGEN_PAPER_X_MAX", "DOMAIN_X_MAX"), float(np.max(coords[:, 0])))
    y_min = _first_config_value(config, ("DILGEN_PAPER_Y_MIN", "DOMAIN_Y_MIN"), float(np.min(coords[:, 1])))
    y_max = _first_config_value(config, ("DILGEN_PAPER_Y_MAX", "DOMAIN_Y_MAX"), float(np.max(coords[:, 1])))
    return float(x_min), float(x_max), float(y_min), float(y_max)


def build_velocity_magnitude_field(velocity, density_space):
    velocity_magnitude = project(sqrt(dot(velocity, velocity)), density_space)
    velocity_magnitude.rename("velocity_magnitude", "velocity_magnitude")
    return velocity_magnitude


def copy_scalar_field(source, name):
    target = Function(source.function_space())
    target.vector().set_local(source.vector().get_local())
    target.vector().apply("insert")
    target.rename(name, name)
    return target


def _cell_measure_values(density_space):
    test = TestFunction(density_space)
    return assemble(test * dx).get_local()


def _write_cell_table(path, density_space, field, active_dofs):
    coords = density_space.tabulate_dof_coordinates().reshape((-1, density_space.mesh().geometry().dim()))
    values = field.vector().get_local()
    areas = _cell_measure_values(density_space)
    active = np.zeros(values.shape[0], dtype=np.int64)
    if active_dofs is not None:
        active[np.asarray(active_dofs, dtype=np.int64)] = 1

    with open(path, "w") as handle:
        handle.write("density_dof\tx\ty\tcell_area\tactive\tvalue\n")
        for dof, value in enumerate(values):
            handle.write(
                "{:d}\t{:.16e}\t{:.16e}\t{:.16e}\t{:d}\t{:.16e}\n".format(
                    dof,
                    float(coords[dof, 0]),
                    float(coords[dof, 1]),
                    float(areas[dof]),
                    int(active[dof]),
                    float(value),
                )
            )


def _evaluate_scalar_field(field, points):
    field.set_allow_extrapolation(True)
    values = np.empty(len(points), dtype=float)
    for idx, point in enumerate(points):
        try:
            values[idx] = float(field(Point(float(point[0]), float(point[1]))))
        except RuntimeError:
            values[idx] = np.nan
    return values


def _write_point_table(path, column_names, rows):
    with open(path, "w") as handle:
        handle.write("\t".join(column_names) + "\n")
        for row in rows:
            handle.write("\t".join("{:.16e}".format(float(value)) for value in row) + "\n")


def _write_grid_table(path, field, x_min, x_max, y_min, y_max, point_count):
    xs = np.linspace(x_min, x_max, point_count)
    ys = np.linspace(y_min, y_max, point_count)
    points = np.asarray([(x, y) for y in ys for x in xs], dtype=float)
    values = _evaluate_scalar_field(field, points)
    rows = np.column_stack((points[:, 0], points[:, 1], values))
    _write_point_table(path, ("x", "y", "value"), rows)


def _write_diagonal_line(path, field, x_min, x_max, y_min, y_max, point_count):
    t = np.linspace(0.0, 1.0, point_count)
    xs = x_min + t * (x_max - x_min)
    ys = y_min + t * (y_max - y_min)
    values = _evaluate_scalar_field(field, np.column_stack((xs, ys)))
    length = math.hypot(x_max - x_min, y_max - y_min)
    rows = np.column_stack((t * length, xs, ys, values))
    _write_point_table(path, ("s", "x", "y", "value"), rows)


def _write_horizontal_line(path, field, x_min, x_max, y_value, point_count):
    xs = np.linspace(x_min, x_max, point_count)
    ys = np.full_like(xs, float(y_value))
    values = _evaluate_scalar_field(field, np.column_stack((xs, ys)))
    rows = np.column_stack((xs, ys, values))
    _write_point_table(path, ("x", "y", "value"), rows)


def _write_readme(output_dir, config, include_g_state):
    full_state_text = (
        "Full wrapper includes the reciprocal wall-distance G state."
        if include_g_state
        else "No reciprocal wall-distance G state is included; this is Dilgen's 3-field SA adjoint."
    )
    with open(os.path.join(output_dir, "README.txt"), "w") as handle:
        handle.write(
            "Dilgen PipeBend paper-style data\n"
            "=================================\n"
            "Files are written once per optimization iteration using the unscaled physical objective gradient dJ/dgamma.\n"
            "The df0dx cell tables are DG0 cell values, i.e. the cell-volume-scaled sensitivities used by the solver.\n"
            "{}\n\n".format(full_state_text)
        )
        handle.write("Contour data:\n")
        handle.write("  velocity_magnitude_cells_###.tsv and velocity_magnitude_grid_###.tsv\n")
        handle.write("  df0dx_cells_###.tsv and df0dx_grid_###.tsv\n\n")
        handle.write("Line data:\n")
        handle.write("  df0dx_diagonal_line_###.tsv: lower-left to upper-right, Dilgen Fig. 5\n")
        handle.write("  df0dx_x_y0p5_line_###.tsv: line (x, 0.5), Dilgen Fig. 6\n\n")
        handle.write("Grid points: {}\n".format(int(config.get("DILGEN_PAPER_GRID_POINTS", 201))))
        handle.write("Line points: {}\n".format(int(config.get("DILGEN_PAPER_LINE_POINTS", 401))))


def write_dilgen_paper_data(
    output_dir,
    iteration,
    velocity_magnitude_field,
    df0dx_field,
    density_space,
    active_dofs,
    config,
    comm,
    include_g_state=False,
):
    if not _is_root(comm):
        return

    os.makedirs(output_dir, exist_ok=True)
    _write_readme(output_dir, config, include_g_state)

    x_min, x_max, y_min, y_max = _domain_bounds(config, density_space)
    grid_points = max(2, int(config.get("DILGEN_PAPER_GRID_POINTS", 201)))
    line_points = max(2, int(config.get("DILGEN_PAPER_LINE_POINTS", 401)))
    horizontal_y = float(config.get("DILGEN_PAPER_HORIZONTAL_LINE_Y", 0.5))
    suffix = "{:03d}".format(int(iteration))

    _write_cell_table(
        os.path.join(output_dir, "velocity_magnitude_cells_{}.tsv".format(suffix)),
        density_space,
        velocity_magnitude_field,
        active_dofs,
    )
    _write_cell_table(
        os.path.join(output_dir, "df0dx_cells_{}.tsv".format(suffix)),
        density_space,
        df0dx_field,
        active_dofs,
    )
    _write_grid_table(
        os.path.join(output_dir, "velocity_magnitude_grid_{}.tsv".format(suffix)),
        velocity_magnitude_field,
        x_min,
        x_max,
        y_min,
        y_max,
        grid_points,
    )
    _write_grid_table(
        os.path.join(output_dir, "df0dx_grid_{}.tsv".format(suffix)),
        df0dx_field,
        x_min,
        x_max,
        y_min,
        y_max,
        grid_points,
    )
    _write_diagonal_line(
        os.path.join(output_dir, "df0dx_diagonal_line_{}.tsv".format(suffix)),
        df0dx_field,
        x_min,
        x_max,
        y_min,
        y_max,
        line_points,
    )
    _write_horizontal_line(
        os.path.join(output_dir, "df0dx_x_y0p5_line_{}.tsv".format(suffix)),
        df0dx_field,
        x_min,
        x_max,
        horizontal_y,
        line_points,
    )
