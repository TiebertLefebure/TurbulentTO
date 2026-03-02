import gmsh

# -----------------------------
# Geometry parameters (meters)
# -----------------------------
MM = 1e-3

R_PIPE = 14.0 * MM      # Pipe radius
R_CURV = 125.0 * MM     # Centerline radius of bend

# Inner/outer radii of the bend walls
R_INNER = R_CURV - R_PIPE
R_OUTER = R_CURV + R_PIPE

D_PIPE = 2.0 * R_PIPE
H_LEG = 30.0 * D_PIPE   # Straight-leg length = 30D = 840 mm

# Medium characteristic element size
LC = D_PIPE / 20.0      # Between coarse (D/10) and fine (D/40) mesh


def create_mesh(output_filename='u_bend_2d.msh'):
    """Generate a medium 2D mesh of a U-bend pipe using Gmsh."""
    gmsh.initialize()
    gmsh.model.add('U_bend_2D_Medium')

    cx, cy = 0.0, 0.0

    def add_point(x, y):
        return gmsh.model.geo.addPoint(x, y, 0.0, LC)

    # Bottom points
    p_out_R_bot = add_point(cx + R_OUTER, cy)
    p_out_L_bot = add_point(cx - R_OUTER, cy)
    p_in_R_bot = add_point(cx + R_INNER, cy)
    p_in_L_bot = add_point(cx - R_INNER, cy)

    # Mid points on semicircle
    p_out_mid = add_point(cx, cy - R_OUTER)
    p_in_mid = add_point(cx, cy - R_INNER)

    # Top of legs
    p_out_R_top = add_point(cx + R_OUTER, cy + H_LEG)
    p_in_R_top = add_point(cx + R_INNER, cy + H_LEG)
    p_out_L_top = add_point(cx - R_OUTER, cy + H_LEG)
    p_in_L_top = add_point(cx - R_INNER, cy + H_LEG)

    p_center = add_point(cx, cy)

    # Boundary curves
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

    # Physical groups
    gmsh.model.addPhysicalGroup(2, [surf], name='Fluid')
    gmsh.model.addPhysicalGroup(1, [L_inlet], name='Inlet')
    gmsh.model.addPhysicalGroup(1, [L_outlet], name='Outlet')
    gmsh.model.addPhysicalGroup(
        1,
        [L_out_left, L_out_right, L_in_left, L_in_right, A_out_1, A_out_2, A_in_1, A_in_2],
        name='Walls',
    )

    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)

    gmsh.finalize()
    print(f'Wrote medium mesh: {output_filename}')


if __name__ == '__main__':
    create_mesh('u_bend_2d.msh')
