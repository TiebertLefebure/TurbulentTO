import argparse
from pathlib import Path

import gmsh
import meshio
import numpy as np


# -----------------------------
# Geometry parameters (meters)
# -----------------------------
MM = 1.0e-3

R_PIPE = 14.0 * MM
R_CURV = 125.0 * MM

R_INNER = R_CURV - R_PIPE
R_OUTER = R_CURV + R_PIPE

D_PIPE = 2.0 * R_PIPE
H_LEG = 30.0 * D_PIPE

# -----------------------------
# Wall-resolved SA target
# -----------------------------
# U-bend SA reference values:
#   U_ref = 1.42 m/s, D_h = 28 mm, nu = 8.9e-7 m^2/s, Re ~= 45,000.
#
# Using Cf = 0.079 Re^(-0.25) gives u_tau ~= 0.074 m/s and
# y_first = nu / u_tau ~= 1.2e-5 m for y+ ~= 1.
FIRST_LAYER_HEIGHT = 1.2e-5

# Physical marker convention used by the U-bend configs:
#   Fluid -> 1, Inlet -> 2, Outlet -> 3, Walls -> 4.
FLUID = 1
INLET = 2
OUTLET = 3
WALLS = 4

MESHES = {
    "Coarse_WallResolved": {
        "bulk": D_PIPE / 10.0,
        "layers": 10,
        "growth": 1.45,
    },
    "Medium_WallResolved": {
        "bulk": D_PIPE / 20.0,
        "layers": 14,
        "growth": 1.30,
    },
    "Fine_WallResolved": {
        "bulk": 8.0e-4,
        "layers": 18,
        "growth": 1.25,
    },
}


def inflation_thickness(first_layer_height, growth, layers):
    return first_layer_height * ((growth ** layers) - 1.0) / (growth - 1.0)


def create_gmsh_mesh(name, settings, msh_file):
    lc_bulk = settings["bulk"]
    layers = settings["layers"]
    growth = settings["growth"]
    thickness = inflation_thickness(FIRST_LAYER_HEIGHT, growth, layers)

    gmsh.initialize()
    try:
        gmsh.model.add(f"U_bend_2D_{name}")

        cx, cy = 0.0, 0.0

        def add_point(x, y):
            return gmsh.model.geo.addPoint(x, y, 0.0, lc_bulk)

        p_out_R_bot = add_point(cx + R_OUTER, cy)
        p_out_L_bot = add_point(cx - R_OUTER, cy)
        p_in_R_bot = add_point(cx + R_INNER, cy)
        p_in_L_bot = add_point(cx - R_INNER, cy)

        p_out_mid = add_point(cx, cy - R_OUTER)
        p_in_mid = add_point(cx, cy - R_INNER)

        p_out_R_top = add_point(cx + R_OUTER, cy + H_LEG)
        p_in_R_top = add_point(cx + R_INNER, cy + H_LEG)
        p_out_L_top = add_point(cx - R_OUTER, cy + H_LEG)
        p_in_L_top = add_point(cx - R_INNER, cy + H_LEG)

        p_center = add_point(cx, cy)

        L_out_left = gmsh.model.geo.addLine(p_out_L_top, p_out_L_bot)
        L_out_right = gmsh.model.geo.addLine(p_out_R_bot, p_out_R_top)
        L_in_right = gmsh.model.geo.addLine(p_in_R_top, p_in_R_bot)
        L_in_left = gmsh.model.geo.addLine(p_in_L_bot, p_in_L_top)

        L_inlet = gmsh.model.geo.addLine(p_in_L_top, p_out_L_top)
        L_outlet = gmsh.model.geo.addLine(p_out_R_top, p_in_R_top)

        A_out_1 = gmsh.model.geo.addCircleArc(p_out_L_bot, p_center, p_out_mid)
        A_out_2 = gmsh.model.geo.addCircleArc(p_out_mid, p_center, p_out_R_bot)
        A_in_1 = gmsh.model.geo.addCircleArc(p_in_R_bot, p_center, p_in_mid)
        A_in_2 = gmsh.model.geo.addCircleArc(p_in_mid, p_center, p_in_L_bot)

        loop = gmsh.model.geo.addCurveLoop([
            L_inlet,
            L_out_left,
            A_out_1,
            A_out_2,
            L_out_right,
            L_outlet,
            L_in_right,
            A_in_1,
            A_in_2,
            L_in_left,
        ])

        surf = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surf], tag=FLUID, name="Fluid")
        gmsh.model.addPhysicalGroup(1, [L_inlet], tag=INLET, name="Inlet")
        gmsh.model.addPhysicalGroup(1, [L_outlet], tag=OUTLET, name="Outlet")

        wall_curves = [
            L_out_left,
            L_out_right,
            L_in_left,
            L_in_right,
            A_out_1,
            A_out_2,
            A_in_1,
            A_in_2,
        ]
        gmsh.model.addPhysicalGroup(1, wall_curves, tag=WALLS, name="Walls")

        fbl = gmsh.model.mesh.field.add("BoundaryLayer")
        gmsh.model.mesh.field.setNumbers(fbl, "CurvesList", wall_curves)
        gmsh.model.mesh.field.setNumbers(
            fbl,
            "FanPointsList",
            [p_out_L_top, p_in_L_top, p_out_R_top, p_in_R_top],
        )
        gmsh.model.mesh.field.setNumber(fbl, "hwall_n", FIRST_LAYER_HEIGHT)
        gmsh.model.mesh.field.setNumber(fbl, "hfar", lc_bulk)
        gmsh.model.mesh.field.setNumber(fbl, "ratio", growth)
        gmsh.model.mesh.field.setNumber(fbl, "NbLayers", layers)
        gmsh.model.mesh.field.setNumber(fbl, "thickness", thickness)
        gmsh.model.mesh.field.setNumber(fbl, "IntersectMetrics", 1)
        gmsh.model.mesh.field.setNumber(fbl, "Quads", 0)
        gmsh.model.mesh.field.setAsBoundaryLayer(fbl)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", FIRST_LAYER_HEIGHT)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_bulk)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh_file))
    finally:
        gmsh.finalize()


def convert_msh_to_xdmf(msh_file):
    msh = meshio.read(msh_file)
    points = msh.points
    if points.shape[1] >= 3:
        max_abs_z = float(np.max(np.abs(points[:, 2])))
        if max_abs_z > 1.0e-12:
            raise RuntimeError(
                f"Mesh appears non-planar (max |z| = {max_abs_z:.3e}); refusing XY projection."
            )
        points = points[:, :2].copy()

    cells = msh.cells_dict
    physical_tags = msh.cell_data_dict["gmsh:physical"]
    output_dir = Path(msh_file).resolve().parent

    triangles = cells["triangle"]
    triangle_tags = physical_tags["triangle"]
    volume_mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        cell_data={"cell_tags": [triangle_tags]},
    )
    meshio.write(output_dir / "mesh.xdmf", volume_mesh)

    lines = cells["line"]
    line_tags = physical_tags["line"]
    facet_mesh = meshio.Mesh(
        points=points,
        cells=[("line", lines)],
        cell_data={"facet_tags": [line_tags]},
    )
    meshio.write(output_dir / "facet.xdmf", facet_mesh)

    return len(points), len(triangles), len(lines)


def generate_mesh(name, settings):
    output_dir = Path(__file__).resolve().parent / name
    output_dir.mkdir(parents=True, exist_ok=True)
    msh_file = output_dir / "u_bend_2d.msh"

    create_gmsh_mesh(name, settings, msh_file)
    vertices, triangles, lines = convert_msh_to_xdmf(msh_file)

    thickness = inflation_thickness(FIRST_LAYER_HEIGHT, settings["growth"], settings["layers"])
    print(
        "{}: vertices={}, triangles={}, boundary_lines={}, first_layer={:.3e} m, "
        "bulk={:.3e} m, layers={}, growth={:.2f}, thickness_per_wall={:.3e} m".format(
            name,
            vertices,
            triangles,
            lines,
            FIRST_LAYER_HEIGHT,
            settings["bulk"],
            settings["layers"],
            settings["growth"],
            thickness,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate all or selected U-bend wall-resolved SA meshes."
    )
    parser.add_argument(
        "mesh_names",
        nargs="*",
        choices=sorted(MESHES),
        help="Optional mesh names to generate. Defaults to all wall-resolved meshes.",
    )
    args = parser.parse_args()

    mesh_names = args.mesh_names or list(MESHES)
    print(
        "U-bend wall-resolved first layer: {:.3e} m (y+ ~= 1)".format(
            FIRST_LAYER_HEIGHT
        )
    )
    for name in mesh_names:
        generate_mesh(name, MESHES[name])


if __name__ == "__main__":
    main()
