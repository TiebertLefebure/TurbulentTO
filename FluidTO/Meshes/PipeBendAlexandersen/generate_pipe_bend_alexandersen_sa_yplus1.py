from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np


WALL_TAG = 1
INLET_TAG = 2
OUTLET_TAG = 3
DESIGN_TAG = 1
NON_DESIGN_FLUID_TAG = 2

L = 1.0
LEAD_LENGTH = 0.2 * L
PORT_HEIGHT = 0.2 * L
INLET_Y_MIN = 0.7 * L
INLET_Y_MAX = INLET_Y_MIN + PORT_HEIGHT
OUTLET_X_MIN = 0.7 * L
OUTLET_X_MAX = OUTLET_X_MIN + PORT_HEIGHT

U_INLET = 1.0
RE_INLET = 5000.0
NU = U_INLET * PORT_HEIGHT / RE_INLET

# Smooth-wall estimate for the SA thesis variant of Alexandersen's case.
SKIN_FRICTION_COEFFICIENT = 0.073 * RE_INLET ** (-0.25)
FRICTION_VELOCITY = U_INLET * np.sqrt(SKIN_FRICTION_COEFFICIENT / 2.0)
Y_PLUS_ONE_DISTANCE = NU / FRICTION_VELOCITY

# The wall-adjacent rectangle is split into two triangles; this keeps the
# farthest centroid in that first strip at approximately y+ = 1.
FIRST_CELL_WIDTH = 1.5 * Y_PLUS_ONE_DISTANCE
BOUNDARY_LAYER_THICKNESS = 0.02
BOUNDARY_LAYER_RATIO = 1.2
H_FAR = 0.007
TOL = 1.0e-12

X_MIN = -LEAD_LENGTH
X_MAX = L
Y_MIN = -LEAD_LENGTH
Y_MAX = L


def unique_sorted(values: list[float], tol: float = TOL) -> np.ndarray:
    out = []
    for value in sorted(values):
        if not out or abs(value - out[-1]) > tol:
            out.append(float(value))
    return np.asarray(out, dtype=float)


def segment_coordinates(a: float, b: float) -> list[float]:
    length = b - a
    if length <= 0.0:
        return [a]

    left = [0.0]
    x = 0.0
    h = FIRST_CELL_WIDTH
    while x + h < min(BOUNDARY_LAYER_THICKNESS, 0.5 * length):
        x += h
        left.append(x)
        h *= BOUNDARY_LAYER_RATIO
    left.append(min(BOUNDARY_LAYER_THICKNESS, 0.5 * length))

    right = [length - value for value in left]
    interior_start = left[-1]
    interior_end = length - left[-1]
    if interior_end > interior_start:
        interior_count = int(np.ceil((interior_end - interior_start) / H_FAR))
        interior = np.linspace(interior_start, interior_end, interior_count + 1).tolist()
    else:
        interior = []

    return [a + value for value in unique_sorted(left + interior + right).tolist()]


def clustered_coordinates(points: tuple[float, ...]) -> np.ndarray:
    coords = []
    sorted_points = sorted(points)
    for a, b in zip(sorted_points[:-1], sorted_points[1:]):
        coords.extend(segment_coordinates(a, b))
    coords.append(sorted_points[-1])
    return unique_sorted(coords)


def node_index(i: int, j: int, ny: int) -> int:
    return i * ny + j


def in_design(x: float, y: float) -> bool:
    return -TOL <= x <= L + TOL and -TOL <= y <= L + TOL


def in_inlet_extension(x: float, y: float) -> bool:
    return (
        X_MIN - TOL <= x <= 0.0 + TOL
        and INLET_Y_MIN - TOL <= y <= INLET_Y_MAX + TOL
    )


def in_outlet_extension(x: float, y: float) -> bool:
    return (
        OUTLET_X_MIN - TOL <= x <= OUTLET_X_MAX + TOL
        and Y_MIN - TOL <= y <= 0.0 + TOL
    )


def in_fluid_domain(x: float, y: float) -> bool:
    return in_design(x, y) or in_inlet_extension(x, y) or in_outlet_extension(x, y)


def classify_boundary_edge(p0: np.ndarray, p1: np.ndarray) -> int:
    midpoint = 0.5 * (p0 + p1)
    x, y = midpoint[:2]
    if abs(x - X_MIN) <= 1.0e-10 and INLET_Y_MIN - TOL <= y <= INLET_Y_MAX + TOL:
        return INLET_TAG
    if abs(y - Y_MIN) <= 1.0e-10 and OUTLET_X_MIN - TOL <= x <= OUTLET_X_MAX + TOL:
        return OUTLET_TAG
    return WALL_TAG


def build_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_coords = clustered_coordinates((X_MIN, 0.0, OUTLET_X_MIN, OUTLET_X_MAX, X_MAX))
    y_coords = clustered_coordinates((Y_MIN, 0.0, INLET_Y_MIN, INLET_Y_MAX, Y_MAX))
    nx = len(x_coords)
    ny = len(y_coords)

    raw_points = np.array([(x, y) for x in x_coords for y in y_coords], dtype=float)
    raw_triangles = []
    raw_tags = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            x_mid = 0.5 * (x_coords[i] + x_coords[i + 1])
            y_mid = 0.5 * (y_coords[j] + y_coords[j + 1])
            if not in_fluid_domain(x_mid, y_mid):
                continue

            n00 = node_index(i, j, ny)
            n10 = node_index(i + 1, j, ny)
            n01 = node_index(i, j + 1, ny)
            n11 = node_index(i + 1, j + 1, ny)
            tag = DESIGN_TAG if in_design(x_mid, y_mid) else NON_DESIGN_FLUID_TAG
            raw_triangles.append((n00, n10, n11))
            raw_triangles.append((n00, n11, n01))
            raw_tags.extend((tag, tag))

    raw_triangles = np.asarray(raw_triangles, dtype=np.int64)
    raw_tags = np.asarray(raw_tags, dtype=np.int32)

    used_nodes = np.unique(raw_triangles.ravel())
    old_to_new = -np.ones(raw_points.shape[0], dtype=np.int64)
    old_to_new[used_nodes] = np.arange(used_nodes.shape[0], dtype=np.int64)
    points = raw_points[used_nodes]
    triangles = old_to_new[raw_triangles]

    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    lines = []
    line_tags = []
    for edge, count in edge_counts.items():
        if count != 1:
            continue
        lines.append(edge)
        line_tags.append(classify_boundary_edge(points[edge[0]], points[edge[1]]))

    return (
        points,
        triangles,
        raw_tags,
        np.asarray(lines, dtype=np.int64),
        np.asarray(line_tags, dtype=np.int32),
    )


def write_mesh(output_dir: Path | str = Path(__file__).resolve().parent) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points, triangles, triangle_tags, lines, line_tags = build_mesh()

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
    cell_path = output_dir / "cell_yplus1.xdmf"
    facet_path = output_dir / "facet_yplus1.xdmf"
    meshio.write(mesh_path, mesh)
    meshio.write(cell_path, mesh)
    meshio.write(facet_path, facets)

    unique_tags, counts = np.unique(triangle_tags, return_counts=True)
    tag_counts = dict(zip(unique_tags.tolist(), counts.tolist()))
    print("Wrote {}".format(mesh_path))
    print("Wrote {}".format(cell_path))
    print("Wrote {}".format(facet_path))
    print("Points: {}".format(points.shape[0]))
    print("Triangles: {}".format(triangles.shape[0]))
    print("Boundary facets: {}".format(lines.shape[0]))
    print("Cell tags: {}".format(tag_counts))
    print("Estimated y+=1 distance: {:.6e} m".format(Y_PLUS_ONE_DISTANCE))
    print("First cell width: {:.6e} m".format(FIRST_CELL_WIDTH))


if __name__ == "__main__":
    write_mesh()
