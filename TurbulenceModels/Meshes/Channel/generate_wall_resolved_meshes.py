import math
from pathlib import Path

import meshio
import numpy as np


# Channel dimensions (meters): periodic streamwise direction x, wall-normal y.
L_CHANNEL = 2.0
H_CHANNEL = 2.0

# Channel SA reference values from ConfigChannel_SpalartAllmaras_Steady.py.
U_REF = 20.0
HYDRAULIC_DIAMETER = 2.0
KINEMATIC_VISCOSITY = 0.00181818
REYNOLDS_NUMBER = U_REF * HYDRAULIC_DIAMETER / KINEMATIC_VISCOSITY

# Estimate y+ ~= 1 first-layer height:
#   Cf = 0.079 Re^(-0.25), u_tau = U_ref sqrt(Cf / 2)
#   y_first = nu / u_tau ~= 1.6e-3 m for Re ~= 22,000.
SKIN_FRICTION_COEFFICIENT = 0.079 * (REYNOLDS_NUMBER ** -0.25)
FRICTION_VELOCITY = U_REF * math.sqrt(SKIN_FRICTION_COEFFICIENT / 2.0)
FIRST_LAYER_HEIGHT = KINEMATIC_VISCOSITY / FRICTION_VELOCITY

# Boundary marker convention used by the existing channel configs:
#   bottom wall -> 1, right/outflow -> 2, top wall -> 3, left/inflow -> 4.
BOTTOM_WALL = 1
OUTFLOW = 2
TOP_WALL = 3
INFLOW = 4

MESHES = {
    "Coarse_WallResolved": {
        "nx": 10,
        "core_dy": 0.08,
        "layers": 10,
        "growth": 1.45,
    },
    "Medium_WallResolved": {
        "nx": 20,
        "core_dy": 0.06,
        "layers": 14,
        "growth": 1.35,
    },
    "Fine_WallResolved": {
        "nx": 40,
        "core_dy": 0.04,
        "layers": 18,
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


def build_y_coordinates(first_layer_height, growth, layers, core_dy):
    bottom = inflation_offsets(first_layer_height, growth, layers)
    boundary_layer_thickness = bottom[-1]
    if 2.0 * boundary_layer_thickness >= H_CHANNEL:
        raise ValueError("Boundary layers overlap; reduce layers/growth.")

    core_min = boundary_layer_thickness
    core_max = H_CHANNEL - boundary_layer_thickness
    core_height = core_max - core_min
    n_core = max(1, int(math.ceil(core_height / core_dy)))
    core = np.linspace(core_min, core_max, n_core + 1).tolist()

    top = [H_CHANNEL - y for y in reversed(bottom)]
    y = bottom[:-1] + core + top[1:]
    return np.array(y, dtype=float)


def build_structured_mesh(nx, y_coordinates):
    x_coordinates = np.linspace(0.0, L_CHANNEL, nx + 1)
    ny = len(y_coordinates) - 1

    points = np.array(
        [(x, y) for y in y_coordinates for x in x_coordinates],
        dtype=float,
    )

    def node(i, j):
        return j * (nx + 1) + i

    triangles = []
    for j in range(ny):
        for i in range(nx):
            n00 = node(i, j)
            n10 = node(i + 1, j)
            n01 = node(i, j + 1)
            n11 = node(i + 1, j + 1)
            triangles.append([n00, n10, n11])
            triangles.append([n00, n11, n01])

    facets = []
    facet_tags = []

    for i in range(nx):
        facets.append([node(i, 0), node(i + 1, 0)])
        facet_tags.append(BOTTOM_WALL)

    for j in range(ny):
        facets.append([node(nx, j), node(nx, j + 1)])
        facet_tags.append(OUTFLOW)

    for i in range(nx):
        facets.append([node(i, ny), node(i + 1, ny)])
        facet_tags.append(TOP_WALL)

    for j in range(ny):
        facets.append([node(0, j), node(0, j + 1)])
        facet_tags.append(INFLOW)

    return (
        points,
        np.array(triangles, dtype=int),
        np.array(facets, dtype=int),
        np.array(facet_tags, dtype=int),
    )


def write_mesh(name, settings):
    output_dir = Path(__file__).resolve().parent / name
    output_dir.mkdir(parents=True, exist_ok=True)

    y_coordinates = build_y_coordinates(
        FIRST_LAYER_HEIGHT,
        settings["growth"],
        settings["layers"],
        settings["core_dy"],
    )
    points, triangles, facets, facet_tags = build_structured_mesh(settings["nx"], y_coordinates)

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

    y_spacing = np.diff(y_coordinates)
    print(
        "{}: vertices={}, triangles={}, nx={}, ny={}, first_layer={:.3e} m, "
        "min_dy={:.3e} m, max_dy={:.3e} m, layers={}, growth={:.2f}".format(
            name,
            len(points),
            len(triangles),
            settings["nx"],
            len(y_coordinates) - 1,
            FIRST_LAYER_HEIGHT,
            float(y_spacing.min()),
            float(y_spacing.max()),
            settings["layers"],
            settings["growth"],
        )
    )


def main():
    print(
        "Channel wall-resolved first layer: {:.3e} m (Re={:.0f}, y+ ~= 1)".format(
            FIRST_LAYER_HEIGHT,
            REYNOLDS_NUMBER,
        )
    )
    for name, settings in MESHES.items():
        write_mesh(name, settings)


if __name__ == "__main__":
    main()
