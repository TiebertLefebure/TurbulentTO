"""Generate a wall-resolved SA mesh for the Alexandersen U-bend benchmark.

Bayat, Li, and Alexandersen use an h_max = 0.007 mesh with implicit
wall-functions for their k-epsilon formulation. This generator keeps that far
field size, but adds y+ <= 1 near-wall refinement for the repository's
wall-resolved Spalart-Allmaras runs.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import gmsh
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
TOP_PORT_Y_MIN = 0.55 * L
TOP_PORT_Y_MAX = TOP_PORT_Y_MIN + PORT_HEIGHT
BOTTOM_PORT_Y_MIN = 0.25 * L
BOTTOM_PORT_Y_MAX = BOTTOM_PORT_Y_MIN + PORT_HEIGHT

BAR_THICKNESS = 0.10 * L
BAR_RADIUS = 0.5 * BAR_THICKNESS
BAR_X_START = -LEAD_LENGTH
BAR_TOTAL_LENGTH = 0.70 * L
BAR_TIP_X = BAR_X_START + BAR_TOTAL_LENGTH
BAR_RECT_X_MAX = BAR_TIP_X - BAR_RADIUS
BAR_Y_MIN = 0.5 * (L - BAR_THICKNESS)

U_INLET = 1.0
RE_INLET = 5000.0
NU = U_INLET * PORT_HEIGHT / RE_INLET

# Smooth-wall estimate for the SA thesis variant of Alexandersen's case.
SKIN_FRICTION_COEFFICIENT = 0.073 * RE_INLET ** (-0.25)
FRICTION_VELOCITY = U_INLET * np.sqrt(SKIN_FRICTION_COEFFICIENT / 2.0)
Y_PLUS_ONE_DISTANCE = NU / FRICTION_VELOCITY

# The near-wall strip is triangulated at this target size; as in the
# Alexandersen pipe-bend mesh, that keeps first-cell centroids near y+ = 1 for
# the SA model.
FIRST_CELL_WIDTH = 1.5 * Y_PLUS_ONE_DISTANCE
WALL_REFINED_THICKNESS = 0.02
H_FAR = 0.007
TOL = 1.0e-9


def between(value: float, lower: float, upper: float, tol: float = TOL) -> bool:
    return (lower - tol) <= value <= (upper + tol)


def external_boundary_curves(surface_tags: list[int]) -> list[int]:
    boundary_pairs = gmsh.model.getBoundary(
        [(2, tag) for tag in surface_tags],
        oriented=False,
        recursive=False,
    )
    counts = Counter(boundary_pairs)
    return [tag for dim, tag in boundary_pairs if dim == 1 and counts[(dim, tag)] == 1]


def classify_surfaces(surface_tags: list[int]) -> tuple[list[int], list[int]]:
    design_surfaces: list[int] = []
    non_design_surfaces: list[int] = []

    for tag in surface_tags:
        x_c, _y_c, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        if x_c < 0.0:
            non_design_surfaces.append(tag)
        else:
            design_surfaces.append(tag)

    return design_surfaces, non_design_surfaces


def classify_boundary_curves(curve_tags: list[int]) -> tuple[list[int], list[int], list[int]]:
    inlet_curves: list[int] = []
    outlet_curves: list[int] = []
    wall_curves: list[int] = []

    for tag in curve_tags:
        x_c, y_c, _ = gmsh.model.occ.getCenterOfMass(1, tag)
        if abs(x_c + LEAD_LENGTH) <= TOL and between(y_c, TOP_PORT_Y_MIN, TOP_PORT_Y_MAX):
            inlet_curves.append(tag)
        elif abs(x_c + LEAD_LENGTH) <= TOL and between(y_c, BOTTOM_PORT_Y_MIN, BOTTOM_PORT_Y_MAX):
            outlet_curves.append(tag)
        else:
            wall_curves.append(tag)

    return wall_curves, inlet_curves, outlet_curves


def build_geometry() -> list[int]:
    occ = gmsh.model.occ
    design = occ.addRectangle(0.0, 0.0, 0.0, L, L)
    inlet = occ.addRectangle(-LEAD_LENGTH, TOP_PORT_Y_MIN, 0.0, LEAD_LENGTH, PORT_HEIGHT)
    outlet = occ.addRectangle(-LEAD_LENGTH, BOTTOM_PORT_Y_MIN, 0.0, LEAD_LENGTH, PORT_HEIGHT)
    fluid, _ = occ.fragment([(2, design)], [(2, inlet), (2, outlet)])

    bar_rect = occ.addRectangle(
        BAR_X_START,
        BAR_Y_MIN,
        0.0,
        BAR_RECT_X_MAX - BAR_X_START,
        BAR_THICKNESS,
    )
    bar_cap = occ.addDisk(BAR_RECT_X_MAX, 0.5 * L, 0.0, BAR_RADIUS, BAR_RADIUS)
    bar, _ = occ.fuse([(2, bar_rect)], [(2, bar_cap)])
    cut, _ = occ.cut(fluid, bar, removeObject=True, removeTool=True)
    return [tag for dim, tag in cut if dim == 2]


def meshio_write_blocks(
    msh_path: Path,
    mesh_xdmf_path: Path,
    facet_xdmf_path: Path,
    cell_xdmf_path: Path,
) -> None:
    msh = meshio.read(msh_path)

    triangle_blocks: list[np.ndarray] = []
    triangle_data: list[np.ndarray] = []
    line_blocks: list[np.ndarray] = []
    line_data: list[np.ndarray] = []

    physical_data = msh.cell_data.get("gmsh:physical", [])
    for cells, data in zip(msh.cells, physical_data):
        if cells.type == "triangle":
            triangle_blocks.append(cells.data)
            triangle_data.append(np.asarray(data, dtype=np.int32))
        elif cells.type == "line":
            line_blocks.append(cells.data)
            line_data.append(np.asarray(data, dtype=np.int32))

    points = msh.points[:, :2]
    triangle_mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", np.vstack(triangle_blocks))],
        cell_data={"name_to_read": [np.concatenate(triangle_data)]},
    )
    facet_mesh = meshio.Mesh(
        points=points,
        cells=[("line", np.vstack(line_blocks))],
        cell_data={"name_to_read": [np.concatenate(line_data)]},
    )

    meshio.write(mesh_xdmf_path, triangle_mesh)
    meshio.write(facet_xdmf_path, facet_mesh)
    meshio.write(cell_xdmf_path, triangle_mesh)


def add_wall_resolved_mesh_field(wall_curves: list[int]) -> None:
    gmsh.option.setNumber("Mesh.MeshSizeMin", FIRST_CELL_WIDTH)
    gmsh.option.setNumber("Mesh.MeshSizeMax", H_FAR)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    wall_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(wall_distance, "CurvesList", wall_curves)
    gmsh.model.mesh.field.setNumber(wall_distance, "Sampling", 200)

    near_wall_size = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(near_wall_size, "InField", wall_distance)
    gmsh.model.mesh.field.setNumber(near_wall_size, "SizeMin", FIRST_CELL_WIDTH)
    gmsh.model.mesh.field.setNumber(near_wall_size, "SizeMax", H_FAR)
    gmsh.model.mesh.field.setNumber(near_wall_size, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(near_wall_size, "DistMax", WALL_REFINED_THICKNESS)
    gmsh.model.mesh.field.setAsBackgroundMesh(near_wall_size)


def generate_case(output_dir: Path | str = Path(__file__).resolve().parent) -> None:
    output_dir = Path(output_dir)

    gmsh.initialize()
    gmsh.model.add("u_bend_alexandersen_sa_yplus1")

    surface_tags = build_geometry()
    gmsh.model.occ.synchronize()

    design_surfaces, non_design_surfaces = classify_surfaces(surface_tags)
    boundary_curves = external_boundary_curves(surface_tags)
    wall_curves, inlet_curves, outlet_curves = classify_boundary_curves(boundary_curves)

    gmsh.model.addPhysicalGroup(2, design_surfaces, DESIGN_TAG)
    gmsh.model.setPhysicalName(2, DESIGN_TAG, "design_domain")
    gmsh.model.addPhysicalGroup(2, non_design_surfaces, NON_DESIGN_FLUID_TAG)
    gmsh.model.setPhysicalName(2, NON_DESIGN_FLUID_TAG, "non_design_fluid")

    gmsh.model.addPhysicalGroup(1, wall_curves, WALL_TAG)
    gmsh.model.setPhysicalName(1, WALL_TAG, "walls")
    gmsh.model.addPhysicalGroup(1, inlet_curves, INLET_TAG)
    gmsh.model.setPhysicalName(1, INLET_TAG, "inlet")
    gmsh.model.addPhysicalGroup(1, outlet_curves, OUTLET_TAG)
    gmsh.model.setPhysicalName(1, OUTLET_TAG, "outlet")

    add_wall_resolved_mesh_field(wall_curves)
    gmsh.model.mesh.generate(2)

    output_dir.mkdir(parents=True, exist_ok=True)
    msh_path = output_dir / "u_bend_alexandersen_sa_yplus1.msh"
    mesh_xdmf_path = output_dir / "mesh_yplus1.xdmf"
    facet_xdmf_path = output_dir / "facet_yplus1.xdmf"
    cell_xdmf_path = output_dir / "cell_yplus1.xdmf"

    gmsh.write(str(msh_path))
    gmsh.finalize()

    meshio_write_blocks(msh_path, mesh_xdmf_path, facet_xdmf_path, cell_xdmf_path)

    msh = meshio.read(msh_path)
    triangle_count = sum(len(block.data) for block in msh.cells if block.type == "triangle")
    print("Wrote {}".format(mesh_xdmf_path))
    print("Wrote {}".format(facet_xdmf_path))
    print("Wrote {}".format(cell_xdmf_path))
    print("Triangles: {}".format(triangle_count))
    print("Estimated y+=1 distance: {:.6e} m".format(Y_PLUS_ONE_DISTANCE))
    print("Near-wall target size: {:.6e} m".format(FIRST_CELL_WIDTH))


if __name__ == "__main__":
    generate_case()
