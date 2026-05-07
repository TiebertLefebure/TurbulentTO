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
NON_DESIGN_SOLID_TAG = 3

H = 0.1
L = 10.0 * H
LEAD_LENGTH = 2.0 * H
PORT_HEIGHT = 2.0 * H
TOP_PORT_Y_MIN = 5.5 * H
TOP_PORT_Y_MAX = TOP_PORT_Y_MIN + PORT_HEIGHT
BOTTOM_PORT_Y_MIN = 2.5 * H
BOTTOM_PORT_Y_MAX = BOTTOM_PORT_Y_MIN + PORT_HEIGHT

BAR_THICKNESS = H
BAR_RADIUS = 0.5 * BAR_THICKNESS
BAR_X_START = -LEAD_LENGTH
BAR_TIP_X = 5.0 * H
BAR_RECT_X_MAX = BAR_TIP_X - BAR_RADIUS
BAR_Y_MIN = 0.5 * (L - BAR_THICKNESS)
BAR_Y_MAX = BAR_Y_MIN + BAR_THICKNESS

U_BULK = 2.0
NU = 4.0e-5
RE_HALF = U_BULK * H / NU

SKIN_FRICTION_COEFFICIENT = 0.073 * RE_HALF ** (-0.25)
FRICTION_VELOCITY = U_BULK * np.sqrt(SKIN_FRICTION_COEFFICIENT / 2.0)
Y_PLUS_ONE_DISTANCE = NU / FRICTION_VELOCITY

# The boundary-layer first strip is split into triangles; this keeps the first
# near-wall centroids below y+ ~= 1 for the Dilgen U-bend inlet conditions.
FIRST_CELL_WIDTH = 1.5 * Y_PLUS_ONE_DISTANCE
# A 0.005 m near-wall layer avoids self-intersection at the rounded separator
# while preserving the y+ ~= 1 first-cell spacing required by the SA model.
BOUNDARY_LAYER_THICKNESS = 0.005
BOUNDARY_LAYER_RATIO = 1.2
H_FAR = 1.0 / 120.0
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


def in_separator_solid(x: float, y: float) -> bool:
    if between(x, BAR_X_START, BAR_RECT_X_MAX) and between(y, BAR_Y_MIN, BAR_Y_MAX):
        return True
    return (x - BAR_RECT_X_MAX) ** 2.0 + (y - 0.5 * L) ** 2.0 <= (BAR_RADIUS + TOL) ** 2.0


def in_lead_solid(x: float, y: float) -> bool:
    if not between(x, -LEAD_LENGTH, 0.0):
        return False
    return y <= BOTTOM_PORT_Y_MIN + TOL or y >= TOP_PORT_Y_MAX - TOL


def classify_surfaces(surface_tags: list[int]) -> tuple[list[int], list[int], list[int]]:
    design_surfaces: list[int] = []
    non_design_fluid_surfaces: list[int] = []
    non_design_solid_surfaces: list[int] = []

    for tag in surface_tags:
        x_c, y_c, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        if in_separator_solid(x_c, y_c) or in_lead_solid(x_c, y_c):
            non_design_solid_surfaces.append(tag)
        elif x_c < 0.0:
            non_design_fluid_surfaces.append(tag)
        else:
            design_surfaces.append(tag)

    return design_surfaces, non_design_fluid_surfaces, non_design_solid_surfaces


def classify_boundary_curves(curve_tags: list[int]) -> tuple[list[int], list[int], list[int]]:
    inlet_curves: list[int] = []
    outlet_curves: list[int] = []
    wall_curves: list[int] = []

    for tag in curve_tags:
        x_c, y_c, _ = gmsh.model.occ.getCenterOfMass(1, tag)
        if abs(x_c + LEAD_LENGTH) <= TOL and between(
            y_c, TOP_PORT_Y_MIN, TOP_PORT_Y_MIN + PORT_HEIGHT
        ):
            inlet_curves.append(tag)
        elif abs(x_c + LEAD_LENGTH) <= TOL and between(
            y_c, BOTTOM_PORT_Y_MIN, BOTTOM_PORT_Y_MIN + PORT_HEIGHT
        ):
            outlet_curves.append(tag)
        else:
            wall_curves.append(tag)

    return wall_curves, inlet_curves, outlet_curves


def build_geometry() -> list[int]:
    occ = gmsh.model.occ
    design = occ.addRectangle(0.0, 0.0, 0.0, L, L)
    inlet = occ.addRectangle(-LEAD_LENGTH, TOP_PORT_Y_MIN, 0.0, LEAD_LENGTH, PORT_HEIGHT)
    outlet = occ.addRectangle(-LEAD_LENGTH, BOTTOM_PORT_Y_MIN, 0.0, LEAD_LENGTH, PORT_HEIGHT)
    bottom_solid = occ.addRectangle(-LEAD_LENGTH, 0.0, 0.0, LEAD_LENGTH, BOTTOM_PORT_Y_MIN)
    top_solid = occ.addRectangle(-LEAD_LENGTH, TOP_PORT_Y_MAX, 0.0, LEAD_LENGTH, L - TOP_PORT_Y_MAX)

    bar_rect = occ.addRectangle(
        BAR_X_START,
        BAR_Y_MIN,
        0.0,
        BAR_RECT_X_MAX - BAR_X_START,
        BAR_THICKNESS,
    )
    bar_cap = occ.addDisk(BAR_RECT_X_MAX, 0.5 * L, 0.0, BAR_RADIUS, BAR_RADIUS)
    bar, _ = occ.fuse([(2, bar_rect)], [(2, bar_cap)])
    design_without_separator, _ = occ.cut(
        [(2, design)],
        bar,
        removeObject=True,
        removeTool=False,
    )

    surfaces = (
        design_without_separator
        + bar
        + [(2, inlet), (2, outlet), (2, bottom_solid), (2, top_solid)]
    )
    fragmented, _ = occ.fragment(surfaces[:1], surfaces[1:], removeObject=True, removeTool=True)
    return [tag for dim, tag in fragmented if dim == 2]


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
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    boundary_layer = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(boundary_layer, "CurvesList", wall_curves)
    gmsh.model.mesh.field.setNumber(boundary_layer, "hwall_n", FIRST_CELL_WIDTH)
    gmsh.model.mesh.field.setNumber(boundary_layer, "thickness", BOUNDARY_LAYER_THICKNESS)
    gmsh.model.mesh.field.setNumber(boundary_layer, "ratio", BOUNDARY_LAYER_RATIO)
    gmsh.model.mesh.field.setNumber(boundary_layer, "Size", H_FAR)
    gmsh.model.mesh.field.setNumber(boundary_layer, "SizeFar", H_FAR)
    gmsh.model.mesh.field.setNumber(boundary_layer, "Quads", 0)
    gmsh.model.mesh.field.setAsBoundaryLayer(boundary_layer)


def generate_case(output_dir: Path | str = Path(__file__).resolve().parent) -> None:
    output_dir = Path(output_dir)

    gmsh.initialize()
    gmsh.model.add("u_bend_dilgen_yplus1")

    surface_tags = build_geometry()
    gmsh.model.occ.synchronize()

    design_surfaces, non_design_fluid_surfaces, non_design_solid_surfaces = classify_surfaces(surface_tags)
    boundary_curves = external_boundary_curves(surface_tags)
    wall_curves, inlet_curves, outlet_curves = classify_boundary_curves(boundary_curves)

    gmsh.model.addPhysicalGroup(2, design_surfaces, DESIGN_TAG)
    gmsh.model.setPhysicalName(2, DESIGN_TAG, "design_domain")
    gmsh.model.addPhysicalGroup(2, non_design_fluid_surfaces, NON_DESIGN_FLUID_TAG)
    gmsh.model.setPhysicalName(2, NON_DESIGN_FLUID_TAG, "non_design_fluid")
    gmsh.model.addPhysicalGroup(2, non_design_solid_surfaces, NON_DESIGN_SOLID_TAG)
    gmsh.model.setPhysicalName(2, NON_DESIGN_SOLID_TAG, "non_design_solid")

    gmsh.model.addPhysicalGroup(1, wall_curves, WALL_TAG)
    gmsh.model.setPhysicalName(1, WALL_TAG, "walls")
    gmsh.model.addPhysicalGroup(1, inlet_curves, INLET_TAG)
    gmsh.model.setPhysicalName(1, INLET_TAG, "inlet")
    gmsh.model.addPhysicalGroup(1, outlet_curves, OUTLET_TAG)
    gmsh.model.setPhysicalName(1, OUTLET_TAG, "outlet")

    add_wall_resolved_mesh_field(wall_curves)
    gmsh.model.mesh.generate(2)

    output_dir.mkdir(parents=True, exist_ok=True)
    msh_path = output_dir / "u_bend_dilgen_yplus1.msh"
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
    print("First boundary-layer width: {:.6e} m".format(FIRST_CELL_WIDTH))


if __name__ == "__main__":
    generate_case()
