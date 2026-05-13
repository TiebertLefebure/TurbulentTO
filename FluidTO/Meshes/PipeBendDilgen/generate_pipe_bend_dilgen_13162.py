#!/usr/bin/env python3
"""Generate a Dilgen channel-bend verification mesh with 13,162 cells.

Dilgen et al. report 13,162 cells for the channel-bend sensitivity
verification case, but do not publish the mesh coordinates. This generator
matches the published geometry, preserves near-wall clustering, includes the
inlet/outlet breakpoints exactly, and writes exactly 13,162 triangular cells
for the FEniCS DG0 design space.
"""

from pathlib import Path

import meshio
import numpy as np


L = 1.0
INLET_WIDTH = 0.2
INLET_TOP_OFFSET = 0.1
OUTLET_WIDTH = 0.2
OUTLET_RIGHT_OFFSET = 0.1

INLET_Y_MAX = L - INLET_TOP_OFFSET
INLET_Y_MIN = INLET_Y_MAX - INLET_WIDTH
OUTLET_X_MAX = L - OUTLET_RIGHT_OFFSET
OUTLET_X_MIN = OUTLET_X_MAX - OUTLET_WIDTH

TARGET_TRIANGLES = 13162
BASE_AXIS_INTERVALS = 81
BOUNDARY_LAYER_WIDTH = 0.02
BOUNDARY_LAYER_INTERVALS = 16
FIRST_CELL_WIDTH = 2.482817665809822e-4
GROWTH_RATE = 1.2
DESIGN_TAG = 1


def geometric_boundary_layer(width, intervals, first_width, growth_rate):
    widths = first_width * growth_rate ** np.arange(intervals, dtype=float)
    return width * widths / np.sum(widths)


def axis_coordinates():
    """Return one clustered axis with exact 0.7 and 0.9 breakpoints."""
    left_widths = geometric_boundary_layer(
        BOUNDARY_LAYER_WIDTH,
        BOUNDARY_LAYER_INTERVALS,
        FIRST_CELL_WIDTH,
        GROWTH_RATE,
    )
    left = np.concatenate(([0.0], np.cumsum(left_widths)))

    # 49 interior intervals: proportional allocation across the three segments
    # split by the inlet/outlet breakpoints.
    middle_1 = np.linspace(BOUNDARY_LAYER_WIDTH, OUTLET_X_MIN, 36)[1:]
    middle_2 = np.linspace(OUTLET_X_MIN, OUTLET_X_MAX, 11)[1:]
    middle_3 = np.linspace(OUTLET_X_MAX, L - BOUNDARY_LAYER_WIDTH, 5)[1:]

    right = L - left[-2::-1]
    coords = np.concatenate((left, middle_1, middle_2, middle_3, right))
    coords = np.unique(np.round(coords, 15))
    if coords.size != BASE_AXIS_INTERVALS + 1:
        raise RuntimeError(
            "Expected {} axis coordinates, got {}.".format(
                BASE_AXIS_INTERVALS + 1,
                coords.size,
            )
        )
    return coords


def build_base_triangles(nx, ny):
    triangles = []
    for j in range(ny):
        row = j * (nx + 1)
        next_row = (j + 1) * (nx + 1)
        for i in range(nx):
            v00 = row + i
            v10 = row + i + 1
            v01 = next_row + i
            v11 = next_row + i + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
    return np.asarray(triangles, dtype=np.int64)


def refine_selected_triangles(points, triangles, refine_count):
    """Split selected interior triangles into three triangles each."""
    centroids = np.mean(points[triangles, :2], axis=1)
    edge_ab = points[triangles[:, 1], :2] - points[triangles[:, 0], :2]
    edge_ac = points[triangles[:, 2], :2] - points[triangles[:, 0], :2]
    areas = 0.5 * np.abs(edge_ab[:, 0] * edge_ac[:, 1] - edge_ab[:, 1] * edge_ac[:, 0])
    distance_to_corner = np.linalg.norm(centroids - np.asarray([0.95, 0.95]), axis=1)
    boundary_margin = np.min(
        np.column_stack(
            (
                centroids[:, 0],
                centroids[:, 1],
                L - centroids[:, 0],
                L - centroids[:, 1],
            )
        ),
        axis=1,
    )
    candidates = np.where(boundary_margin > 1.0e-3)[0]
    order = candidates[np.lexsort((-areas[candidates], distance_to_corner[candidates]))]
    selected = set(int(idx) for idx in order[:refine_count])

    refined_points = points.tolist()
    refined_triangles = []
    for tri_idx, tri in enumerate(triangles):
        if tri_idx not in selected:
            refined_triangles.append(tri.tolist())
            continue
        centroid = np.mean(points[tri], axis=0)
        centroid_idx = len(refined_points)
        refined_points.append(centroid.tolist())
        a, b, c = tri.tolist()
        refined_triangles.extend(
            [
                [a, b, centroid_idx],
                [b, c, centroid_idx],
                [c, a, centroid_idx],
            ]
        )

    return np.asarray(refined_points, dtype=float), np.asarray(refined_triangles, dtype=np.int64)


def main():
    output_dir = Path(__file__).resolve().parent
    x_coords = axis_coordinates()
    y_coords = axis_coordinates()
    points = np.asarray([(x, y) for y in y_coords for x in x_coords], dtype=float)

    base_triangles = build_base_triangles(len(x_coords) - 1, len(y_coords) - 1)
    missing_triangles = TARGET_TRIANGLES - base_triangles.shape[0]
    if missing_triangles < 0 or missing_triangles % 2:
        raise RuntimeError("Target triangle count is incompatible with centroid refinement.")
    points, triangles = refine_selected_triangles(points, base_triangles, missing_triangles // 2)
    if triangles.shape[0] != TARGET_TRIANGLES:
        raise RuntimeError(
            "Expected {} triangles, got {}.".format(TARGET_TRIANGLES, triangles.shape[0])
        )

    mesh = meshio.Mesh(
        points,
        [("triangle", triangles)],
        cell_data={"name_to_read": [np.full(triangles.shape[0], DESIGN_TAG, dtype=np.int32)]},
    )
    meshio.write(output_dir / "mesh_dilgen2018_13162.xdmf", mesh)
    print(
        "Wrote {} with {} points and {} triangles.".format(
            output_dir / "mesh_dilgen2018_13162.xdmf",
            points.shape[0],
            triangles.shape[0],
        )
    )


if __name__ == "__main__":
    main()
