import gmsh

# ---------------------------------
# Geometry parameters
# ---------------------------------
L_X = 1.0
L_Y = 1.0

# Port layout (must match Config_PipeBendBorrvall_Laminar.py)
INLET_WIDTH      = 0.2
INLET_TOP_OFFSET = 0.2   # distance from top
OUTLET_WIDTH      = 0.2
OUTLET_RIGHT_OFFSET = 0.2  # distance from right

INLET_Y_MAX  = L_Y - INLET_TOP_OFFSET          # 0.8
INLET_Y_MIN  = INLET_Y_MAX - INLET_WIDTH        # 0.6
OUTLET_X_MAX = L_X - OUTLET_RIGHT_OFFSET        # 0.8
OUTLET_X_MIN = OUTLET_X_MAX - OUTLET_WIDTH      # 0.6

# Uniform mesh size matched to N = 120 in the config
LC = L_X / 120.0


def create_mesh(output_filename='pipe_bend_2d.msh'):
    gmsh.initialize()
    gmsh.model.add('PipeBend_2D')

    # Points counterclockwise from bottom-left
    p1  = gmsh.model.geo.addPoint(0.0,         0.0,        0.0, LC)  # bottom-left
    p2  = gmsh.model.geo.addPoint(OUTLET_X_MIN, 0.0,        0.0, LC)  # outlet left on bottom
    p3  = gmsh.model.geo.addPoint(OUTLET_X_MAX, 0.0,        0.0, LC)  # outlet right on bottom
    p4  = gmsh.model.geo.addPoint(L_X,          0.0,        0.0, LC)  # bottom-right
    p5  = gmsh.model.geo.addPoint(L_X,          L_Y,        0.0, LC)  # top-right
    p6  = gmsh.model.geo.addPoint(0.0,          L_Y,        0.0, LC)  # top-left
    p7  = gmsh.model.geo.addPoint(0.0,          INLET_Y_MAX, 0.0, LC) # inlet top on left
    p8  = gmsh.model.geo.addPoint(0.0,          INLET_Y_MIN, 0.0, LC) # inlet bottom on left

    L_bot_left   = gmsh.model.geo.addLine(p1, p2)  # wall
    L_outlet     = gmsh.model.geo.addLine(p2, p3)  # outlet
    L_bot_right  = gmsh.model.geo.addLine(p3, p4)  # wall
    L_right      = gmsh.model.geo.addLine(p4, p5)  # wall
    L_top        = gmsh.model.geo.addLine(p5, p6)  # wall
    L_left_top   = gmsh.model.geo.addLine(p6, p7)  # wall above inlet
    L_inlet      = gmsh.model.geo.addLine(p7, p8)  # inlet
    L_left_bot   = gmsh.model.geo.addLine(p8, p1)  # wall below inlet

    loop = gmsh.model.geo.addCurveLoop([
        L_bot_left, L_outlet, L_bot_right, L_right,
        L_top, L_left_top, L_inlet, L_left_bot,
    ])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surf], name='Fluid')
    gmsh.model.addPhysicalGroup(1, [L_inlet], name='Inlet')
    gmsh.model.addPhysicalGroup(1, [L_outlet], name='Outlet')
    wall_curves = [L_bot_left, L_bot_right, L_right, L_top, L_left_top, L_left_bot]
    gmsh.model.addPhysicalGroup(1, wall_curves, name='Walls')

    gmsh.option.setNumber('Mesh.CharacteristicLengthMin', LC)
    gmsh.option.setNumber('Mesh.CharacteristicLengthMax', LC)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)

    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)
    gmsh.finalize()

    print(f'Wrote pipe bend mesh: {output_filename}')


if __name__ == '__main__':
    create_mesh('pipe_bend_2d.msh')
