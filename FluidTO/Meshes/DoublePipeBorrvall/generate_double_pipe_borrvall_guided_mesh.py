from __future__ import annotations

from collections import Counter
from pathlib import Path

import gmsh
import meshio
import numpy as np


WALL_TAG = 1
INLET_TAG = 2
OUTLET_TAG = 3
DOUBLE_PIPE_INLET_TAGS = (2, 3)   # top, bottom
DOUBLE_PIPE_OUTLET_TAGS = (4, 5)  # top, bottom

DESIGN_TAG = 1
NON_DESIGN_FLUID_TAG = 2

L = 1.0
N = 120
LC = L / N
LEAD_WIDTH = 0.2 * L
TOL = 1.0e-9


def between(value: float, lower: float, upper: float, tol: float = TOL) -> bool:
    return (lower - tol) <= value <= (upper + tol)


def classify_surfaces(case_name: str, surface_tags: list[int]) -> tuple[list[int], list[int]]:
    design_surfaces: list[int] = []
    non_design_surfaces: list[int] = []

    for tag in surface_tags:
        x_c, y_c, _ = gmsh.model.occ.getCenterOfMass(2, tag)

        if case_name == "diffuser":
            if x_c < 0.0 or x_c > L:
                non_design_surfaces.append(tag)
            else:
                design_surfaces.append(tag)
        elif case_name == "double_pipe":
            if x_c < 0.0 or x_c > 1.5 * L:
                non_design_surfaces.append(tag)
            else:
                design_surfaces.append(tag)
        elif case_name == "pipe_bend":
            if x_c < 0.0 or y_c < 0.0 or x_c > L or y_c > L:
                non_design_surfaces.append(tag)
            else:
                design_surfaces.append(tag)
        else:
            raise ValueError(f"Unknown case name: {case_name}")

    return design_surfaces, non_design_surfaces


def external_boundary_curves(surface_tags: list[int]) -> list[int]:
    boundary_pairs = gmsh.model.getBoundary([(2, tag) for tag in surface_tags], oriented=False, recursive=False)
    counts = Counter(boundary_pairs)
    return [tag for dim, tag in boundary_pairs if dim == 1 and counts[(dim, tag)] == 1]


def classify_boundary_curves(case_name: str, curve_tags: list[int]) -> tuple[list[int], list[int], list[int]]:
    inlet_curves: list[int] = []
    outlet_curves: list[int] = []
    wall_curves: list[int] = []

    for tag in curve_tags:
        x_c, y_c, _ = gmsh.model.occ.getCenterOfMass(1, tag)

        if case_name == "diffuser":
            outlet_y_min = L / 3.0
            outlet_y_max = 2.0 * L / 3.0
            if abs(x_c + LEAD_WIDTH) <= TOL and between(y_c, 0.0, L):
                inlet_curves.append(tag)
            elif abs(x_c - (L + LEAD_WIDTH)) <= TOL and between(y_c, outlet_y_min, outlet_y_max):
                outlet_curves.append(tag)
            else:
                wall_curves.append(tag)
        elif case_name == "double_pipe":
            x_out = 1.5 * L + LEAD_WIDTH
            openings = (
                (L / 6.0, L / 3.0),
                (2.0 * L / 3.0, 5.0 * L / 6.0),
            )
            if abs(x_c + LEAD_WIDTH) <= TOL and any(between(y_c, y0, y1) for y0, y1 in openings):
                inlet_curves.append(tag)
            elif abs(x_c - x_out) <= TOL and any(between(y_c, y0, y1) for y0, y1 in openings):
                outlet_curves.append(tag)
            else:
                wall_curves.append(tag)
        elif case_name == "pipe_bend":
            inlet_y_min = 0.6 * L
            inlet_y_max = 0.8 * L
            outlet_x_min = 0.6 * L
            outlet_x_max = 0.8 * L
            if abs(x_c + LEAD_WIDTH) <= TOL and between(y_c, inlet_y_min, inlet_y_max):
                inlet_curves.append(tag)
            elif abs(y_c + LEAD_WIDTH) <= TOL and between(x_c, outlet_x_min, outlet_x_max):
                outlet_curves.append(tag)
            else:
                wall_curves.append(tag)
        else:
            raise ValueError(f"Unknown case name: {case_name}")

    return wall_curves, inlet_curves, outlet_curves


def _sort_curves_top_to_bottom(curve_tags: list[int]) -> list[int]:
    return sorted(
        curve_tags,
        key=lambda tag: gmsh.model.occ.getCenterOfMass(1, tag)[1],
        reverse=True,
    )


def add_boundary_physical_groups(case_name: str, wall_curves: list[int], inlet_curves: list[int], outlet_curves: list[int]) -> None:
    gmsh.model.addPhysicalGroup(1, wall_curves, WALL_TAG)
    gmsh.model.setPhysicalName(1, WALL_TAG, "walls")

    if case_name == "double_pipe":
        inlet_curves_sorted = _sort_curves_top_to_bottom(inlet_curves)
        outlet_curves_sorted = _sort_curves_top_to_bottom(outlet_curves)
        if len(inlet_curves_sorted) != len(DOUBLE_PIPE_INLET_TAGS):
            raise ValueError(f"Expected two double-pipe inlet curves, got {len(inlet_curves_sorted)}.")
        if len(outlet_curves_sorted) != len(DOUBLE_PIPE_OUTLET_TAGS):
            raise ValueError(f"Expected two double-pipe outlet curves, got {len(outlet_curves_sorted)}.")

        for marker, curve, name in zip(DOUBLE_PIPE_INLET_TAGS, inlet_curves_sorted, ("inlet_top", "inlet_bottom")):
            gmsh.model.addPhysicalGroup(1, [curve], marker)
            gmsh.model.setPhysicalName(1, marker, name)
        for marker, curve, name in zip(DOUBLE_PIPE_OUTLET_TAGS, outlet_curves_sorted, ("outlet_top", "outlet_bottom")):
            gmsh.model.addPhysicalGroup(1, [curve], marker)
            gmsh.model.setPhysicalName(1, marker, name)
        return

    gmsh.model.addPhysicalGroup(1, inlet_curves, INLET_TAG)
    gmsh.model.setPhysicalName(1, INLET_TAG, "inlet")
    gmsh.model.addPhysicalGroup(1, outlet_curves, OUTLET_TAG)
    gmsh.model.setPhysicalName(1, OUTLET_TAG, "outlet")


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


def build_diffuser_geometry() -> None:
    design = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)
    inlet = gmsh.model.occ.addRectangle(-LEAD_WIDTH, 0.0, 0.0, LEAD_WIDTH, L)
    outlet = gmsh.model.occ.addRectangle(L, L / 3.0, 0.0, LEAD_WIDTH, L / 3.0)
    gmsh.model.occ.fragment([(2, design)], [(2, inlet), (2, outlet)])


def build_double_pipe_geometry() -> None:
    design = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.5 * L, L)
    leads = [
        gmsh.model.occ.addRectangle(-LEAD_WIDTH, L / 6.0, 0.0, LEAD_WIDTH, L / 6.0),
        gmsh.model.occ.addRectangle(-LEAD_WIDTH, 2.0 * L / 3.0, 0.0, LEAD_WIDTH, L / 6.0),
        gmsh.model.occ.addRectangle(1.5 * L, L / 6.0, 0.0, LEAD_WIDTH, L / 6.0),
        gmsh.model.occ.addRectangle(1.5 * L, 2.0 * L / 3.0, 0.0, LEAD_WIDTH, L / 6.0),
    ]
    gmsh.model.occ.fragment([(2, design)], [(2, lead) for lead in leads])


def build_pipe_bend_geometry() -> None:
    design = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)
    inlet = gmsh.model.occ.addRectangle(-LEAD_WIDTH, 0.6 * L, 0.0, LEAD_WIDTH, 0.2 * L)
    outlet = gmsh.model.occ.addRectangle(0.6 * L, -LEAD_WIDTH, 0.0, 0.2 * L, LEAD_WIDTH)
    gmsh.model.occ.fragment([(2, design)], [(2, inlet), (2, outlet)])


def generate_case(case_name: str, out_dir: Path) -> None:
    gmsh.initialize()
    gmsh.model.add(case_name)

    if case_name == "diffuser":
        build_diffuser_geometry()
        msh_name = "diffuser_guided_2d.msh"
    elif case_name == "double_pipe":
        build_double_pipe_geometry()
        msh_name = "double_pipe_guided_2d.msh"
    elif case_name == "pipe_bend":
        build_pipe_bend_geometry()
        msh_name = "pipe_bend_guided_2d.msh"
    else:
        raise ValueError(f"Unknown case name: {case_name}")

    gmsh.model.occ.synchronize()

    surface_tags = [tag for _, tag in gmsh.model.getEntities(2)]
    design_surfaces, non_design_surfaces = classify_surfaces(case_name, surface_tags)
    boundary_curves = external_boundary_curves(surface_tags)
    wall_curves, inlet_curves, outlet_curves = classify_boundary_curves(case_name, boundary_curves)

    gmsh.model.addPhysicalGroup(2, design_surfaces, DESIGN_TAG)
    gmsh.model.setPhysicalName(2, DESIGN_TAG, "design_domain")
    gmsh.model.addPhysicalGroup(2, non_design_surfaces, NON_DESIGN_FLUID_TAG)
    gmsh.model.setPhysicalName(2, NON_DESIGN_FLUID_TAG, "non_design_fluid")

    add_boundary_physical_groups(case_name, wall_curves, inlet_curves, outlet_curves)

    gmsh.option.setNumber("Mesh.MeshSizeMin", LC)
    gmsh.option.setNumber("Mesh.MeshSizeMax", LC)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    msh_path = out_dir / msh_name
    mesh_xdmf_path = out_dir / "mesh_guided.xdmf"
    facet_xdmf_path = out_dir / "facet_guided.xdmf"
    cell_xdmf_path = out_dir / "cell_guided.xdmf"

    gmsh.write(str(msh_path))
    gmsh.finalize()

    meshio_write_blocks(msh_path, mesh_xdmf_path, facet_xdmf_path, cell_xdmf_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    generate_case("double_pipe", repo_root / "Meshes" / "DoublePipeBorrvall")


if __name__ == "__main__":
    main()
