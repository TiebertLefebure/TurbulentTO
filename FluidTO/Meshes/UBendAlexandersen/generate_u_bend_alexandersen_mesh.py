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
PIPE_BEND_INLET_Y_MIN = 0.7 * L
PIPE_BEND_OUTLET_X_MIN = 0.7 * L
U_BEND_TOP_PORT_Y_MIN = 0.55 * L
U_BEND_BOTTOM_PORT_Y_MIN = 0.25 * L
U_BEND_BAR_THICKNESS = 0.10 * L
U_BEND_BAR_RADIUS = 0.5 * U_BEND_BAR_THICKNESS
U_BEND_BAR_TOTAL_LENGTH = 0.70 * L
U_BEND_BAR_TIP_X = -LEAD_LENGTH + U_BEND_BAR_TOTAL_LENGTH
U_BEND_BAR_RECT_X_MAX = U_BEND_BAR_TIP_X - U_BEND_BAR_RADIUS
H_MAX = 0.007
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


def classify_pipe_bend_surfaces(surface_tags: list[int]) -> tuple[list[int], list[int]]:
    design_surfaces: list[int] = []
    non_design_surfaces: list[int] = []

    for tag in surface_tags:
        x_c, y_c, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        if x_c < 0.0 or y_c < 0.0:
            non_design_surfaces.append(tag)
        else:
            design_surfaces.append(tag)

    return design_surfaces, non_design_surfaces


def classify_u_bend_surfaces(surface_tags: list[int]) -> tuple[list[int], list[int]]:
    design_surfaces: list[int] = []
    non_design_surfaces: list[int] = []

    for tag in surface_tags:
        x_c, _y_c, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        if x_c < 0.0:
            non_design_surfaces.append(tag)
        else:
            design_surfaces.append(tag)

    return design_surfaces, non_design_surfaces


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


def classify_pipe_bend_boundary_curves(curve_tags: list[int]) -> tuple[list[int], list[int], list[int]]:
    inlet_curves: list[int] = []
    outlet_curves: list[int] = []
    wall_curves: list[int] = []

    for tag in curve_tags:
        x_c, y_c, _ = gmsh.model.occ.getCenterOfMass(1, tag)
        if abs(x_c + LEAD_LENGTH) <= TOL and between(
            y_c, PIPE_BEND_INLET_Y_MIN, PIPE_BEND_INLET_Y_MIN + PORT_HEIGHT
        ):
            inlet_curves.append(tag)
        elif abs(y_c + LEAD_LENGTH) <= TOL and between(
            x_c, PIPE_BEND_OUTLET_X_MIN, PIPE_BEND_OUTLET_X_MIN + PORT_HEIGHT
        ):
            outlet_curves.append(tag)
        else:
            wall_curves.append(tag)

    return wall_curves, inlet_curves, outlet_curves


def classify_u_bend_boundary_curves(curve_tags: list[int]) -> tuple[list[int], list[int], list[int]]:
    inlet_curves: list[int] = []
    outlet_curves: list[int] = []
    wall_curves: list[int] = []

    for tag in curve_tags:
        x_c, y_c, _ = gmsh.model.occ.getCenterOfMass(1, tag)
        if abs(x_c + LEAD_LENGTH) <= TOL and between(
            y_c, U_BEND_TOP_PORT_Y_MIN, U_BEND_TOP_PORT_Y_MIN + PORT_HEIGHT
        ):
            inlet_curves.append(tag)
        elif abs(x_c + LEAD_LENGTH) <= TOL and between(
            y_c, U_BEND_BOTTOM_PORT_Y_MIN, U_BEND_BOTTOM_PORT_Y_MIN + PORT_HEIGHT
        ):
            outlet_curves.append(tag)
        else:
            wall_curves.append(tag)

    return wall_curves, inlet_curves, outlet_curves


def build_pipe_bend_geometry() -> list[int]:
    occ = gmsh.model.occ
    design = occ.addRectangle(0.0, 0.0, 0.0, L, L)
    inlet = occ.addRectangle(-LEAD_LENGTH, PIPE_BEND_INLET_Y_MIN, 0.0, LEAD_LENGTH, PORT_HEIGHT)
    outlet = occ.addRectangle(PIPE_BEND_OUTLET_X_MIN, -LEAD_LENGTH, 0.0, PORT_HEIGHT, LEAD_LENGTH)
    occ.fragment([(2, design)], [(2, inlet), (2, outlet)])
    return [tag for dim, tag in occ.getEntities(2)]


def build_u_bend_geometry() -> list[int]:
    occ = gmsh.model.occ
    design = occ.addRectangle(0.0, 0.0, 0.0, L, L)
    inlet = occ.addRectangle(-LEAD_LENGTH, U_BEND_TOP_PORT_Y_MIN, 0.0, LEAD_LENGTH, PORT_HEIGHT)
    outlet = occ.addRectangle(-LEAD_LENGTH, U_BEND_BOTTOM_PORT_Y_MIN, 0.0, LEAD_LENGTH, PORT_HEIGHT)
    fluid, _ = occ.fragment([(2, design)], [(2, inlet), (2, outlet)])

    bar_rect = occ.addRectangle(
        -LEAD_LENGTH,
        0.5 * (L - U_BEND_BAR_THICKNESS),
        0.0,
        U_BEND_BAR_RECT_X_MAX + LEAD_LENGTH,
        U_BEND_BAR_THICKNESS,
    )
    bar_cap = occ.addDisk(U_BEND_BAR_RECT_X_MAX, 0.5 * L, 0.0, U_BEND_BAR_RADIUS, U_BEND_BAR_RADIUS)
    bar, _ = occ.fuse([(2, bar_rect)], [(2, bar_cap)])
    cut, _ = occ.cut(fluid, bar, removeObject=True, removeTool=True)
    return [tag for dim, tag in cut if dim == 2]


def generate_case(
    case_name: str,
    out_dir: Path,
    geometry_builder,
    surface_classifier,
    boundary_classifier,
    msh_name: str,
) -> None:
    gmsh.initialize()
    gmsh.model.add(case_name)

    surface_tags = geometry_builder()
    gmsh.model.occ.synchronize()

    design_surfaces, non_design_surfaces = surface_classifier(surface_tags)
    boundary_curves = external_boundary_curves(surface_tags)
    wall_curves, inlet_curves, outlet_curves = boundary_classifier(boundary_curves)

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

    gmsh.option.setNumber("Mesh.MeshSizeMin", H_MAX)
    gmsh.option.setNumber("Mesh.MeshSizeMax", H_MAX)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    msh_path = out_dir / msh_name
    mesh_xdmf_path = out_dir / "mesh_guided.xdmf"
    facet_xdmf_path = out_dir / "facet_guided.xdmf"
    cell_xdmf_path = out_dir / "cell_guided.xdmf"

    gmsh.write(str(msh_path))
    gmsh.finalize()

    meshio_write_blocks(msh_path, mesh_xdmf_path, facet_xdmf_path, cell_xdmf_path)

    msh = meshio.read(msh_path)
    triangle_count = sum(
        len(block.data)
        for block in msh.cells
        if block.type == "triangle"
    )
    print(f"{case_name}: wrote {triangle_count} triangles to {msh_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    generate_case(
        "u_bend_alexandersen",
        repo_root / "Meshes" / "UBendAlexandersen",
        build_u_bend_geometry,
        classify_u_bend_surfaces,
        classify_u_bend_boundary_curves,
        "u_bend_alexandersen_2d.msh",
    )


if __name__ == "__main__":
    main()
