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
# Mesh controls
# -----------------------------
# Bulk element size: same as Ansys_Inflation (D/35 ≈ 0.8 mm), keeps the core
# mesh comparable between runs.
LC_BULK = 8.0e-4  # D_PIPE / 35

# Wall-resolved inflation parameters.
# Target y+ ≈ 1 at Re = 45 000 (U_ref = 1.42 m/s, D_h = 28 mm, nu = 8.9e-7 m^2/s).
#
#   u_tau = U_ref * sqrt(Cf/2),  Cf = 0.079 * Re^(-0.25)
#   Re    = 1.42 * 0.028 / 8.9e-7 ≈ 4.5e4
#   Cf    = 0.079 * (4.5e4)^(-0.25) ≈ 0.0054
#   u_tau ≈ 1.42 * sqrt(0.0027) ≈ 0.074 m/s
#
#   y_first = y+ * nu / u_tau = 1.0 * 8.9e-7 / 0.074 ≈ 1.2e-5 m
#
# 18 layers with growth 1.25:
#   last layer   ≈ 1.2e-5 * 1.25^17 ≈ 3.4e-4 m  (ratio to bulk ≈ 0.43, smooth)
#   total BL thickness per wall ≈ 1.66 mm  (12% of D_PIPE, well within pipe interior)
#   estimated element count ≈ 300 000–400 000  (well below 500 000 RAM limit)
N_INFLATION_LAYERS = 18
INFLATION_GROWTH = 1.25
FIRST_LAYER_HEIGHT = 1.2e-5  # m, gives y+ ≈ 1

TOTAL_INFLATION_THICKNESS = FIRST_LAYER_HEIGHT * (
    (INFLATION_GROWTH ** N_INFLATION_LAYERS) - 1.0
) / (INFLATION_GROWTH - 1.0)

USE_QUADS_IN_INFLATION = False


def create_mesh(output_filename='u_bend_2d.msh'):
    gmsh.initialize()
    gmsh.model.add('U_bend_2D_Fine_WallRefinement')

    cx, cy = 0.0, 0.0

    def add_point(x, y):
        return gmsh.model.geo.addPoint(x, y, 0.0, LC_BULK)

    p_out_R_bot = add_point(cx + R_OUTER, cy)
    p_out_L_bot = add_point(cx - R_OUTER, cy)
    p_in_R_bot  = add_point(cx + R_INNER, cy)
    p_in_L_bot  = add_point(cx - R_INNER, cy)

    p_out_mid = add_point(cx, cy - R_OUTER)
    p_in_mid  = add_point(cx, cy - R_INNER)

    p_out_R_top = add_point(cx + R_OUTER, cy + H_LEG)
    p_in_R_top  = add_point(cx + R_INNER, cy + H_LEG)
    p_out_L_top = add_point(cx - R_OUTER, cy + H_LEG)
    p_in_L_top  = add_point(cx - R_INNER, cy + H_LEG)

    p_center = add_point(cx, cy)

    L_out_left  = gmsh.model.geo.addLine(p_out_L_top, p_out_L_bot)
    L_out_right = gmsh.model.geo.addLine(p_out_R_bot, p_out_R_top)
    L_in_right  = gmsh.model.geo.addLine(p_in_R_top, p_in_R_bot)
    L_in_left   = gmsh.model.geo.addLine(p_in_L_bot, p_in_L_top)

    L_inlet  = gmsh.model.geo.addLine(p_in_L_top, p_out_L_top)
    L_outlet = gmsh.model.geo.addLine(p_out_R_top, p_in_R_top)

    A_out_1 = gmsh.model.geo.addCircleArc(p_out_L_bot, p_center, p_out_mid)
    A_out_2 = gmsh.model.geo.addCircleArc(p_out_mid,   p_center, p_out_R_bot)
    A_in_1  = gmsh.model.geo.addCircleArc(p_in_R_bot,  p_center, p_in_mid)
    A_in_2  = gmsh.model.geo.addCircleArc(p_in_mid,    p_center, p_in_L_bot)

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

    # Physical groups — must be added in this order so tags match the config:
    #   Fluid  → tag 1 (2D)
    #   Inlet  → tag 2 (1D)
    #   Outlet → tag 3 (1D)
    #   Walls  → tag 4 (1D)
    gmsh.model.addPhysicalGroup(2, [surf],     name='Fluid')
    gmsh.model.addPhysicalGroup(1, [L_inlet],  name='Inlet')
    gmsh.model.addPhysicalGroup(1, [L_outlet], name='Outlet')

    wall_curves = [L_out_left, L_out_right, L_in_left, L_in_right,
                   A_out_1, A_out_2, A_in_1, A_in_2]
    gmsh.model.addPhysicalGroup(1, wall_curves, name='Walls')

    # True inflation layers (anisotropic BL) — same field API as Ansys_Inflation.
    fbl = gmsh.model.mesh.field.add('BoundaryLayer')
    gmsh.model.mesh.field.setNumbers(fbl, 'CurvesList', wall_curves)
    gmsh.model.mesh.field.setNumbers(
        fbl,
        'FanPointsList',
        [p_out_L_top, p_in_L_top, p_out_R_top, p_in_R_top],
    )
    gmsh.model.mesh.field.setNumber(fbl, 'hwall_n',   FIRST_LAYER_HEIGHT)
    gmsh.model.mesh.field.setNumber(fbl, 'hfar',      LC_BULK)
    gmsh.model.mesh.field.setNumber(fbl, 'ratio',     INFLATION_GROWTH)
    gmsh.model.mesh.field.setNumber(fbl, 'NbLayers',  N_INFLATION_LAYERS)
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

    print(f'Wrote wall-resolved inflation mesh: {output_filename}')
    print(
        'Inflation settings: '
        f'layers={N_INFLATION_LAYERS}, '
        f'first_layer={FIRST_LAYER_HEIGHT:.3e} m  (y+ ≈ 1), '
        f'growth={INFLATION_GROWTH:.3f}, '
        f'thickness_per_wall={TOTAL_INFLATION_THICKNESS:.3e} m, '
        f'bulk={LC_BULK:.3e} m'
    )


if __name__ == '__main__':
    create_mesh('u_bend_2d.msh')
