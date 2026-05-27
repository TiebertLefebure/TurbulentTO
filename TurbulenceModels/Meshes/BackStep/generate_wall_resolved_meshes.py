import argparse
import math
from pathlib import Path

import meshio
import numpy as np


# Geometry used by the legacy BackStep meshes.
X_INLET = -32.5
X_WALL_START = -27.5
X_STEP = 0.0
X_OUTLET = 12.5
Y_BOTTOM = -0.25
Y_UPSTREAM_BOTTOM = 0.0
Y_TOP = 2.0

STEP_HEIGHT = Y_UPSTREAM_BOTTOM - Y_BOTTOM
INLET_HEIGHT = Y_TOP - Y_UPSTREAM_BOTTOM

# SA backstep operating point from ConfigBackStep_SpalartAllmaras_Steady.py.
U_REF = 25.0
KINEMATIC_VISCOSITY = 0.000181818
REYNOLDS_NUMBER_STEP = U_REF * STEP_HEIGHT / KINEMATIC_VISCOSITY
REYNOLDS_NUMBER_INLET_HEIGHT = U_REF * INLET_HEIGHT / KINEMATIC_VISCOSITY

# Conservative y+ ~= 1 estimate using the step-height Reynolds number.
SKIN_FRICTION_COEFFICIENT = 0.079 * (REYNOLDS_NUMBER_STEP ** -0.25)
FRICTION_VELOCITY = U_REF * math.sqrt(SKIN_FRICTION_COEFFICIENT / 2.0)
TARGET_FIRST_LAYER_HEIGHT = KINEMATIC_VISCOSITY / FRICTION_VELOCITY
FIRST_LAYER_HEIGHT = TARGET_FIRST_LAYER_HEIGHT
FIRST_LAYER_Y_PLUS = FIRST_LAYER_HEIGHT / TARGET_FIRST_LAYER_HEIGHT

# Boundary marker convention used by the backstep configs:
#   lower wall + vertical step -> 1, outflow -> 2, top wall -> 3,
#   inflow -> 4, upstream slip/symmetry segments -> 5.
WALLS_LOWER_AND_STEP = 1
OUTFLOW = 2
TOP_WALL = 3
INFLOW = 4
SYMMETRY = 5

MESHES = {
    "Coarse": {
        "upstream_dx": 0.20,
        "downstream_dx": 0.10,
        "core_dy_upper": 0.08,
        "core_dy_lower": 0.025,
        "layers": 14,
        "growth": 1.35,
    },
    "Medium": {
        "upstream_dx": 0.12,
        "downstream_dx": 0.065,
        "core_dy_upper": 0.055,
        "core_dy_lower": 0.018,
        "layers": 18,
        "growth": 1.30,
    },
    "Fine": {
        "upstream_dx": 0.08,
        "downstream_dx": 0.04,
        "core_dy_upper": 0.04,
        "core_dy_lower": 0.012,
        "layers": 22,
        "growth": 1.25,
    },
}


def inflation_offsets(first_layer_height, growth, layers):
    offsets = [0.0]
    spacing = first_layer_height
    for _ in range(layers):
        offsets.append(offsets[-1] + spacing)
        spacing *= growth
    return offsets


def line_coordinates(start, end, target_spacing):
    length = abs(end - start)
    if length <= 0.0:
        return np.array([start], dtype=float)
    count = max(1, int(math.ceil(length / target_spacing)))
    return np.linspace(start, end, count + 1)


def append_unique(values, new_values, tolerance=1.0e-12):
    for value in new_values:
        if not values or abs(float(value) - values[-1]) > tolerance:
            values.append(float(value))
    return values


def build_inflated_interval(start, end, first_layer_height, growth, layers, core_spacing):
    offsets = inflation_offsets(first_layer_height, growth, layers)
    thickness = offsets[-1]
    if 2.0 * thickness >= (end - start):
        raise ValueError("Boundary layers overlap for interval [{}, {}].".format(start, end))

    lower = [start + offset for offset in offsets]
    core_start = start + thickness
    core_end = end - thickness
    core = line_coordinates(core_start, core_end, core_spacing)
    upper = [end - offset for offset in reversed(offsets)]

    values = []
    append_unique(values, lower)
    append_unique(values, core[1:])
    append_unique(values, upper[1:])
    return np.array(values, dtype=float)


def build_x_coordinates(settings):
    downstream_offsets = inflation_offsets(
        FIRST_LAYER_HEIGHT,
        settings["growth"],
        settings["layers"],
    )
    x_values = []
    append_unique(x_values, line_coordinates(X_INLET, X_WALL_START, settings["upstream_dx"]))
    append_unique(x_values, line_coordinates(X_WALL_START, X_STEP, settings["upstream_dx"])[1:])

    # Keep a wall-normal x-inflation from the vertical step wall into the
    # downstream region.
    x_near_step = [X_STEP + offset for offset in downstream_offsets[1:]]
    append_unique(x_values, x_near_step)

    last_inflation_x = x_values[-1]
    append_unique(
        x_values,
        line_coordinates(last_inflation_x, X_OUTLET, settings["downstream_dx"])[1:],
    )
    return np.array(x_values, dtype=float)


def build_y_coordinates(settings):
    y_lower = build_inflated_interval(
        Y_BOTTOM,
        Y_UPSTREAM_BOTTOM,
        FIRST_LAYER_HEIGHT,
        settings["growth"],
        settings["layers"],
        settings["core_dy_lower"],
    )
    y_upper = build_inflated_interval(
        Y_UPSTREAM_BOTTOM,
        Y_TOP,
        FIRST_LAYER_HEIGHT,
        settings["growth"],
        settings["layers"],
        settings["core_dy_upper"],
    )
    return np.concatenate([y_lower[:-1], y_upper])


def inside_domain(x, y, tolerance=1.0e-12):
    in_upper = (
        x >= X_INLET - tolerance
        and x <= X_OUTLET + tolerance
        and y >= Y_UPSTREAM_BOTTOM - tolerance
        and y <= Y_TOP + tolerance
    )
    in_lower_downstream = (
        x >= X_STEP - tolerance
        and x <= X_OUTLET + tolerance
        and y >= Y_BOTTOM - tolerance
        and y <= Y_UPSTREAM_BOTTOM + tolerance
    )
    return in_upper or in_lower_downstream


def build_structured_backstep_mesh(settings):
    x_coordinates = build_x_coordinates(settings)
    y_coordinates = build_y_coordinates(settings)

    points = []
    node_lookup = {}

    def node(i, j):
        key = (i, j)
        if key in node_lookup:
            return node_lookup[key]
        x = float(x_coordinates[i])
        y = float(y_coordinates[j])
        if not inside_domain(x, y):
            return None
        node_lookup[key] = len(points)
        points.append((x, y))
        return node_lookup[key]

    triangles = []
    for j in range(len(y_coordinates) - 1):
        y_mid = 0.5 * (y_coordinates[j] + y_coordinates[j + 1])
        for i in range(len(x_coordinates) - 1):
            x_mid = 0.5 * (x_coordinates[i] + x_coordinates[i + 1])
            if not inside_domain(float(x_mid), float(y_mid)):
                continue

            n00 = node(i, j)
            n10 = node(i + 1, j)
            n01 = node(i, j + 1)
            n11 = node(i + 1, j + 1)
            if None in (n00, n10, n01, n11):
                continue
            triangles.append([n00, n10, n11])
            triangles.append([n00, n11, n01])

    facets = []
    facet_tags = []

    def add_facet(n0, n1, tag):
        if n0 is None or n1 is None:
            return
        facets.append([n0, n1])
        facet_tags.append(tag)

    y_index_step = int(np.where(np.isclose(y_coordinates, Y_UPSTREAM_BOTTOM))[0][0])
    y_index_top = int(np.where(np.isclose(y_coordinates, Y_TOP))[0][0])
    y_index_bottom = int(np.where(np.isclose(y_coordinates, Y_BOTTOM))[0][0])

    for i in range(len(x_coordinates) - 1):
        x_mid = 0.5 * (x_coordinates[i] + x_coordinates[i + 1])

        if X_INLET <= x_mid <= X_STEP:
            tag = SYMMETRY if x_mid < X_WALL_START else WALLS_LOWER_AND_STEP
            add_facet(
                node(i, y_index_step),
                node(i + 1, y_index_step),
                tag,
            )

        if X_INLET <= x_mid <= X_OUTLET:
            tag = SYMMETRY if x_mid < X_WALL_START else TOP_WALL
            add_facet(
                node(i, y_index_top),
                node(i + 1, y_index_top),
                tag,
            )

        if X_STEP <= x_mid <= X_OUTLET:
            add_facet(node(i, y_index_bottom), node(i + 1, y_index_bottom), WALLS_LOWER_AND_STEP)

    x_index_inlet = int(np.where(np.isclose(x_coordinates, X_INLET))[0][0])
    x_index_step = int(np.where(np.isclose(x_coordinates, X_STEP))[0][0])
    x_index_outlet = int(np.where(np.isclose(x_coordinates, X_OUTLET))[0][0])

    for j in range(y_index_step, y_index_top):
        add_facet(node(x_index_inlet, j), node(x_index_inlet, j + 1), INFLOW)

    for j in range(y_index_bottom, y_index_top):
        add_facet(node(x_index_outlet, j), node(x_index_outlet, j + 1), OUTFLOW)

    for j in range(y_index_bottom, y_index_step):
        add_facet(node(x_index_step, j), node(x_index_step, j + 1), WALLS_LOWER_AND_STEP)

    return (
        np.array(points, dtype=float),
        np.array(triangles, dtype=int),
        np.array(facets, dtype=int),
        np.array(facet_tags, dtype=int),
        x_coordinates,
        y_coordinates,
    )


def write_mesh(name, settings):
    output_dir = Path(__file__).resolve().parent / name
    output_dir.mkdir(parents=True, exist_ok=True)

    points, triangles, facets, facet_tags, x_coordinates, y_coordinates = build_structured_backstep_mesh(settings)
    cell_tags = np.zeros(len(triangles), dtype=int)

    volume_mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        cell_data={"name_to_read": [cell_tags]},
    )
    facet_mesh = meshio.Mesh(
        points=points,
        cells=[("line", facets)],
        cell_data={"name_to_read": [facet_tags]},
    )

    meshio.write(output_dir / "mesh.xdmf", volume_mesh)
    meshio.write(output_dir / "facet.xdmf", facet_mesh)

    y_index_step = int(np.where(np.isclose(y_coordinates, Y_UPSTREAM_BOTTOM))[0][0])
    dy_lower = np.diff(y_coordinates[: y_index_step + 1])
    dy_upper = np.diff(y_coordinates[y_index_step:])
    dx_downstream = np.diff(x_coordinates[x_coordinates >= X_STEP])

    print(
        "{}: vertices={}, triangles={}, boundary_lines={}, first_layer={:.3e} m, "
        "estimated_y+={:.2f}, min_dx_downstream={:.3e} m, min_dy_lower={:.3e} m, "
        "min_dy_upper={:.3e} m, layers={}, growth={:.2f}".format(
            name,
            len(points),
            len(triangles),
            len(facets),
            FIRST_LAYER_HEIGHT,
            FIRST_LAYER_Y_PLUS,
            float(dx_downstream.min()),
            float(dy_lower.min()),
            float(dy_upper.min()),
            settings["layers"],
            settings["growth"],
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate all or selected wall-resolved backward-facing-step meshes."
    )
    parser.add_argument(
        "mesh_names",
        nargs="*",
        choices=sorted(MESHES),
        help="Optional mesh names to generate. Defaults to all backstep meshes.",
    )
    args = parser.parse_args()

    mesh_names = args.mesh_names or list(MESHES)
    print(
        "BackStep wall-resolved first layer: {:.3e} m "
        "(Re_h={:.0f}, Re_Hin={:.0f}, u_tau={:.3f} m/s, target y+=1 height={:.3e} m, y+={:.2f})".format(
            FIRST_LAYER_HEIGHT,
            REYNOLDS_NUMBER_STEP,
            REYNOLDS_NUMBER_INLET_HEIGHT,
            FRICTION_VELOCITY,
            TARGET_FIRST_LAYER_HEIGHT,
            FIRST_LAYER_Y_PLUS,
        )
    )
    for name in mesh_names:
        write_mesh(name, MESHES[name])


if __name__ == "__main__":
    main()
