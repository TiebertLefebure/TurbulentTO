from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np


WALL_TAG = 1
INLET_TAG = 2
OUTLET_TAG = 3

L = 1.0
INLET_WIDTH = 0.2
INLET_TOP_OFFSET = 0.1
OUTLET_WIDTH = 0.2
OUTLET_RIGHT_OFFSET = 0.1

INLET_Y_MAX = L - INLET_TOP_OFFSET
INLET_Y_MIN = INLET_Y_MAX - INLET_WIDTH
OUTLET_X_MAX = L - OUTLET_RIGHT_OFFSET
OUTLET_X_MIN = OUTLET_X_MAX - OUTLET_WIDTH

U_BULK = 5.0
NU = 5.0e-5
H_HALF = 0.1
RE_HALF = U_BULK * H_HALF / NU

# Smooth-wall turbulent-channel estimate. This gives y ~= 1.66e-4 m for y+ = 1.
SKIN_FRICTION_COEFFICIENT = 0.073 * RE_HALF ** (-0.25)
FRICTION_VELOCITY = U_BULK * np.sqrt(SKIN_FRICTION_COEFFICIENT / 2.0)
Y_PLUS_ONE_DISTANCE = NU / FRICTION_VELOCITY

# The wall-adjacent rectangle is split into two triangles; this keeps the
# farthest centroid in that first strip at approximately y+ = 1.
FIRST_CELL_WIDTH = 1.5 * Y_PLUS_ONE_DISTANCE
BOUNDARY_LAYER_THICKNESS = 0.02
BOUNDARY_LAYER_RATIO = 1.2
H_FAR = 1.0 / 120.0
TOL = 1.0e-12


def unique_sorted(values: list[float], tol: float = TOL) -> np.ndarray:
    out = []
    for value in sorted(values):
        if not out or abs(value - out[-1]) > tol:
            out.append(float(value))
    return np.asarray(out, dtype=float)


def clustered_axis_coordinates(length: float, fixed_points: tuple[float, ...]) -> np.ndarray:
    left = [0.0]
    x = 0.0
    h = FIRST_CELL_WIDTH
    while x + h < BOUNDARY_LAYER_THICKNESS:
        x += h
        left.append(x)
        h *= BOUNDARY_LAYER_RATIO
    left.append(BOUNDARY_LAYER_THICKNESS)

    right = [length - value for value in left]

    interior_start = BOUNDARY_LAYER_THICKNESS
    interior_end = length - BOUNDARY_LAYER_THICKNESS
    interior_count = int(np.ceil((interior_end - interior_start) / H_FAR))
    interior = np.linspace(interior_start, interior_end, interior_count + 1).tolist()

    return unique_sorted(left + interior + right + list(fixed_points) + [length])


def node_index(i: int, j: int, ny: int) -> int:
    return i * ny + j


def build_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_coords = clustered_axis_coordinates(L, (OUTLET_X_MIN, OUTLET_X_MAX))
    y_coords = clustered_axis_coordinates(L, (INLET_Y_MIN, INLET_Y_MAX))
    nx = len(x_coords)
    ny = len(y_coords)

    points = np.array([(x, y) for x in x_coords for y in y_coords], dtype=float)

    triangles = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            n00 = node_index(i, j, ny)
            n10 = node_index(i + 1, j, ny)
            n01 = node_index(i, j + 1, ny)
            n11 = node_index(i + 1, j + 1, ny)
            triangles.append((n00, n10, n11))
            triangles.append((n00, n11, n01))

    lines = []
    line_tags = []

    for i in range(nx - 1):
        x_mid = 0.5 * (x_coords[i] + x_coords[i + 1])
        tag = OUTLET_TAG if OUTLET_X_MIN - TOL <= x_mid <= OUTLET_X_MAX + TOL else WALL_TAG
        lines.append((node_index(i, 0, ny), node_index(i + 1, 0, ny)))
        line_tags.append(tag)
        lines.append((node_index(i, ny - 1, ny), node_index(i + 1, ny - 1, ny)))
        line_tags.append(WALL_TAG)

    for j in range(ny - 1):
        y_mid = 0.5 * (y_coords[j] + y_coords[j + 1])
        tag = INLET_TAG if INLET_Y_MIN - TOL <= y_mid <= INLET_Y_MAX + TOL else WALL_TAG
        lines.append((node_index(0, j, ny), node_index(0, j + 1, ny)))
        line_tags.append(tag)
        lines.append((node_index(nx - 1, j, ny), node_index(nx - 1, j + 1, ny)))
        line_tags.append(WALL_TAG)

    return (
        points,
        np.asarray(triangles, dtype=np.int64),
        np.asarray(lines, dtype=np.int64),
        np.asarray(line_tags, dtype=np.int32),
    )


def write_mesh(output_dir: Path | str = Path(__file__).resolve().parent) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points, triangles, lines, line_tags = build_mesh()
    triangle_tags = np.ones(triangles.shape[0], dtype=np.int32)

    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        cell_data={"name_to_read": [triangle_tags]},
    )
    facets = meshio.Mesh(
        points=points,
        cells=[("line", lines)],
        cell_data={"name_to_read": [line_tags]},
    )

    mesh_path = output_dir / "mesh_yplus1.xdmf"
    facet_path = output_dir / "facet_yplus1.xdmf"
    meshio.write(mesh_path, mesh)
    meshio.write(facet_path, facets)

    print("Wrote {}".format(mesh_path))
    print("Wrote {}".format(facet_path))
    print("Points: {}".format(points.shape[0]))
    print("Triangles: {}".format(triangles.shape[0]))
    print("Boundary facets: {}".format(lines.shape[0]))
    print("Estimated y+=1 distance: {:.6e} m".format(Y_PLUS_ONE_DISTANCE))
    print("First cell width: {:.6e} m".format(FIRST_CELL_WIDTH))


if __name__ == "__main__":
    write_mesh()
