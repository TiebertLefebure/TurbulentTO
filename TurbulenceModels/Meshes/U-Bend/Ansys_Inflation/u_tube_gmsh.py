import gmsh

# -----------------------------
# Geometry parameters (meters)
# -----------------------------
MM = 1e-3

R_PIPE = 14.0 * MM
R_CURV = 125.0 * MM

R_INNER = R_CURV - R_PIPE
R_OUTER = R_CURV + R_PIPE

D_PIPE = 2.0 * R_PIPE
H_LEG = 30.0 * D_PIPE

# -----------------------------
# Mesh controls (ANSYS-like target)
# -----------------------------
# Bulk element size requested by user benchmark.
LC_BULK = 8.0e-4 # LC_BULK = D_PIPE / 35


# Boundary-layer inflation controls.
N_INFLATION_LAYERS = 6
INFLATION_GROWTH = 1.25

# Set first layer so that outer inflation layer is close to LC_BULK.
FIRST_LAYER_HEIGHT = LC_BULK / (INFLATION_GROWTH ** (N_INFLATION_LAYERS - 1))
TOTAL_INFLATION_THICKNESS = FIRST_LAYER_HEIGHT * (
    (INFLATION_GROWTH ** N_INFLATION_LAYERS) - 1.0
) / (INFLATION_GROWTH - 1.0)

# Legacy FEniCS pipeline in this repo expects triangles; keep False by default.
USE_QUADS_IN_INFLATION = False


def create_mesh(output_filename='u_bend_2d.msh'):
    gmsh.initialize()
    gmsh.model.add('U_bend_2D_Ansys_Inflation')

    cx, cy = 0.0, 0.0

    def add_point(x, y):
        return gmsh.model.geo.addPoint(x, y, 0.0, LC_BULK)

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

    gmsh.model.addPhysicalGroup(2, [surf], name='Fluid')
    gmsh.model.addPhysicalGroup(1, [L_inlet], name='Inlet')
    gmsh.model.addPhysicalGroup(1, [L_outlet], name='Outlet')

    wall_curves = [L_out_left, L_out_right, L_in_left, L_in_right, A_out_1, A_out_2, A_in_1, A_in_2]
    gmsh.model.addPhysicalGroup(1, wall_curves, name='Walls')

    # True inflation layers (anisotropic boundary layer), instead of pure size-thresholding.
    fbl = gmsh.model.mesh.field.add('BoundaryLayer')
    gmsh.model.mesh.field.setNumbers(fbl, 'CurvesList', wall_curves)
    gmsh.model.mesh.field.setNumbers(
        fbl,
        'FanPointsList',
        [p_out_L_top, p_in_L_top, p_out_R_top, p_in_R_top],
    )
    gmsh.model.mesh.field.setNumber(fbl, 'hwall_n', FIRST_LAYER_HEIGHT)
    gmsh.model.mesh.field.setNumber(fbl, 'hfar', LC_BULK)
    gmsh.model.mesh.field.setNumber(fbl, 'ratio', INFLATION_GROWTH)
    gmsh.model.mesh.field.setNumber(fbl, 'NbLayers', N_INFLATION_LAYERS)
    gmsh.model.mesh.field.setNumber(fbl, 'thickness', TOTAL_INFLATION_THICKNESS)
    gmsh.model.mesh.field.setNumber(fbl, 'IntersectMetrics', 1)
    gmsh.model.mesh.field.setNumber(fbl, 'Quads', 1 if USE_QUADS_IN_INFLATION else 0)
    gmsh.model.mesh.field.setAsBoundaryLayer(fbl)

    gmsh.option.setNumber('Mesh.CharacteristicLengthMin', FIRST_LAYER_HEIGHT)
    gmsh.option.setNumber('Mesh.CharacteristicLengthMax', LC_BULK)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)

    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)
    gmsh.finalize()

    print(f'Wrote fine inflation mesh: {output_filename}')
    print(
        'Inflation settings: '
        f'layers={N_INFLATION_LAYERS}, '
        f'first_layer={FIRST_LAYER_HEIGHT:.3e} m, '
        f'growth={INFLATION_GROWTH:.3f}, '
        f'thickness={TOTAL_INFLATION_THICKNESS:.3e} m, '
        f'bulk={LC_BULK:.3e} m, '
        f'quads={"on" if USE_QUADS_IN_INFLATION else "off"}'
    )


if __name__ == '__main__':
    create_mesh('u_bend_2d.msh')
