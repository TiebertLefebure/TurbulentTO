from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np


WALL_TAG = 1
INLET_LEFT_TOP_TAG = 2
INLET_RIGHT_BOTTOM_TAG = 3
OUTLET_RIGHT_TOP_TAG = 4
OUTLET_LEFT_BOTTOM_TAG = 5
DESIGN_TAG = 1

L1 = 3.0
L2 = 2.0
H1 = 5.0
H2 = 1.0
H3 = 1.0

X_MIN = -L2
DESIGN_X_MIN = 0.0
DESIGN_X_MAX = L1
X_MAX = L1 + L2
Y_MIN = 0.0
Y_MAX = H1

BOTTOM_INLET_Y_MIN = 1.0
BOTTOM_INLET_Y_MAX = BOTTOM_INLET_Y_MIN + H2
TOP_INLET_Y_MIN = 3.0
TOP_INLET_Y_MAX = TOP_INLET_Y_MIN + H2

BOTTOM_OUTLET_Y_MIN = 1.0
BOTTOM_OUTLET_Y_MAX = BOTTOM_OUTLET_Y_MIN + H3
TOP_OUTLET_Y_MIN = 3.0
TOP_OUTLET_Y_MAX = TOP_OUTLET_Y_MIN + H3

RHO = 1.0
MU = 1.0
NU = MU / RHO
U_MAX = 3000.0
PORT_HEIGHT = H2

RE_WALL = RHO * U_MAX * PORT_HEIGHT / MU
SKIN_FRICTION_COEFFICIENT = 0.073 * RE_WALL ** (-0.25)
FRICTION_VELOCITY = U_MAX * np.sqrt(SKIN_FRICTION_COEFFICIENT / 2.0)
Y_PLUS_ONE_DISTANCE = NU / FRICTION_VELOCITY

N = 50
H_FAR = 1.0 / N
FIRST_CELL_WIDTH = min(H_FAR, Y_PLUS_ONE_DISTANCE)
BOUNDARY_LAYER_THICKNESS = 0.10
BOUNDARY_LAYER_RATIO = 1.20
TOL = 1.0e-12


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

    layer = min(BOUNDARY_LAYER_THICKNESS, 0.5 * length)
    left = [0.0]
    x = 0.0
    h = FIRST_CELL_WIDTH
    while x + h < layer:
        x += h
        left.append(x)
        h *= BOUNDARY_LAYER_RATIO
    left.append(layer)

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


def between(value: float, lower: float, upper: float) -> bool:
    return lower - TOL <= value <= upper + TOL


def in_segment(value: float, segment: tuple[float, float]) -> bool:
    return between(value, segment[0], segment[1])


LEFT_TOP_SEGMENT = (TOP_INLET_Y_MIN, TOP_INLET_Y_MAX)
LEFT_BOTTOM_SEGMENT = (BOTTOM_OUTLET_Y_MIN, BOTTOM_OUTLET_Y_MAX)
RIGHT_TOP_SEGMENT = (TOP_OUTLET_Y_MIN, TOP_OUTLET_Y_MAX)
RIGHT_BOTTOM_SEGMENT = (BOTTOM_INLET_Y_MIN, BOTTOM_INLET_Y_MAX)

LEFT_PORT_SEGMENTS = (LEFT_TOP_SEGMENT, LEFT_BOTTOM_SEGMENT)
RIGHT_PORT_SEGMENTS = (RIGHT_TOP_SEGMENT, RIGHT_BOTTOM_SEGMENT)


def in_left_port(y: float) -> bool:
    return any(in_segment(y, segment) for segment in LEFT_PORT_SEGMENTS)


def in_right_port(y: float) -> bool:
    return any(in_segment(y, segment) for segment in RIGHT_PORT_SEGMENTS)


def in_design(x: float, y: float) -> bool:
    return between(x, DESIGN_X_MIN, DESIGN_X_MAX) and between(y, Y_MIN, Y_MAX)


def in_left_extension(x: float, y: float) -> bool:
    return between(x, X_MIN, DESIGN_X_MIN) and in_left_port(y)


def in_right_extension(x: float, y: float) -> bool:
    return between(x, DESIGN_X_MAX, X_MAX) and in_right_port(y)


def in_fluid_domain(x: float, y: float) -> bool:
    return in_design(x, y) or in_left_extension(x, y) or in_right_extension(x, y)


def classify_boundary_edge(p0: np.ndarray, p1: np.ndarray) -> int:
    midpoint = 0.5 * (p0 + p1)
    x, y = midpoint[:2]
    if abs(x - X_MIN) <= 1.0e-10:
        if in_segment(y, LEFT_TOP_SEGMENT):
            return INLET_LEFT_TOP_TAG
        if in_segment(y, LEFT_BOTTOM_SEGMENT):
            return OUTLET_LEFT_BOTTOM_TAG
    if abs(x - X_MAX) <= 1.0e-10:
        if in_segment(y, RIGHT_TOP_SEGMENT):
            return OUTLET_RIGHT_TOP_TAG
        if in_segment(y, RIGHT_BOTTOM_SEGMENT):
            return INLET_RIGHT_BOTTOM_TAG
    return WALL_TAG


def build_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_coords = clustered_coordinates((X_MIN, DESIGN_X_MIN, DESIGN_X_MAX, X_MAX))
    y_coords = clustered_coordinates(
        (
            Y_MIN,
            BOTTOM_INLET_Y_MIN,
            BOTTOM_INLET_Y_MAX,
            TOP_INLET_Y_MIN,
            TOP_INLET_Y_MAX,
            Y_MAX,
        )
    )
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
            raw_triangles.append((n00, n10, n11))
            raw_triangles.append((n00, n11, n01))
            raw_tags.extend((DESIGN_TAG, DESIGN_TAG))

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

    mesh_path = output_dir / "mesh_yoon_yplus1.xdmf"
    cell_path = output_dir / "cell_yoon_yplus1.xdmf"
    facet_path = output_dir / "facet_yoon_yplus1.xdmf"
    meshio.write(mesh_path, mesh)
    meshio.write(cell_path, mesh)
    meshio.write(facet_path, facets)

    unique_tags, counts = np.unique(triangle_tags, return_counts=True)
    marker_tags, marker_counts = np.unique(line_tags, return_counts=True)
    print("Wrote {}".format(mesh_path))
    print("Wrote {}".format(cell_path))
    print("Wrote {}".format(facet_path))
    print("Points: {}".format(points.shape[0]))
    print("Triangles: {}".format(triangles.shape[0]))
    print("Boundary facets: {}".format(lines.shape[0]))
    print("Cell tags: {}".format(dict(zip(unique_tags.tolist(), counts.tolist()))))
    print("Boundary tags: {}".format(dict(zip(marker_tags.tolist(), marker_counts.tolist()))))
    print("Re_wall: {:.6e}".format(RE_WALL))
    print("Estimated y+=1 distance: {:.6e} m".format(Y_PLUS_ONE_DISTANCE))
    print("First cell width: {:.6e} m".format(FIRST_CELL_WIDTH))


if __name__ == "__main__":
    write_mesh()
