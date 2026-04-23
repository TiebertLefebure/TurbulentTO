import gmsh

# ---------------------------------
# Geometry parameters
# ---------------------------------
L_X = 1.5
L_Y = 1.0

# Port layout (must match Config_DoublePipeBorrvall_LaminarTO.py)
PORT_WIDTH         = 1.0 / 6.0
PORT_TOP_MARGIN    = 1.0 / 4.0
PORT_BOTTOM_MARGIN = 1.0 / 4.0

TOP_PORT_Y_MAX    = L_Y - PORT_TOP_MARGIN            # 0.75
TOP_PORT_Y_MIN    = TOP_PORT_Y_MAX - PORT_WIDTH       # 7/12
BOTTOM_PORT_Y_MIN = PORT_BOTTOM_MARGIN                # 0.25
BOTTOM_PORT_Y_MAX = BOTTOM_PORT_Y_MIN + PORT_WIDTH    # 5/12

# Uniform mesh size matched to NX=150, NY=100 in the config
LC = min(L_X / 150.0, L_Y / 100.0) # LC = 0.01 m


def create_mesh(output_filename='double_pipe_2d.msh'):
    gmsh.initialize()
    gmsh.model.add('DoublePipe_2D')

    # Points counterclockwise from bottom-left.
    # Left wall split at port y-values; right wall split at the same y-values.
    p1  = gmsh.model.geo.addPoint(0.0, 0.0,               0.0, LC)  # bottom-left
    p2  = gmsh.model.geo.addPoint(L_X, 0.0,               0.0, LC)  # bottom-right
    p3  = gmsh.model.geo.addPoint(L_X, BOTTOM_PORT_Y_MIN, 0.0, LC)  # right: below bottom port
    p4  = gmsh.model.geo.addPoint(L_X, BOTTOM_PORT_Y_MAX, 0.0, LC)  # right: above bottom port
    p5  = gmsh.model.geo.addPoint(L_X, TOP_PORT_Y_MIN,    0.0, LC)  # right: below top port
    p6  = gmsh.model.geo.addPoint(L_X, TOP_PORT_Y_MAX,    0.0, LC)  # right: above top port
    p7  = gmsh.model.geo.addPoint(L_X, L_Y,               0.0, LC)  # top-right
    p8  = gmsh.model.geo.addPoint(0.0, L_Y,               0.0, LC)  # top-left
    p9  = gmsh.model.geo.addPoint(0.0, TOP_PORT_Y_MAX,    0.0, LC)  # left: above top port
    p10 = gmsh.model.geo.addPoint(0.0, TOP_PORT_Y_MIN,    0.0, LC)  # left: below top port
    p11 = gmsh.model.geo.addPoint(0.0, BOTTOM_PORT_Y_MAX, 0.0, LC)  # left: above bottom port
    p12 = gmsh.model.geo.addPoint(0.0, BOTTOM_PORT_Y_MIN, 0.0, LC)  # left: below bottom port

    L_bottom         = gmsh.model.geo.addLine(p1,  p2)   # wall
    L_right_1        = gmsh.model.geo.addLine(p2,  p3)   # wall
    L_outlet_bottom  = gmsh.model.geo.addLine(p3,  p4)   # bottom outlet
    L_right_2        = gmsh.model.geo.addLine(p4,  p5)   # wall between ports
    L_outlet_top     = gmsh.model.geo.addLine(p5,  p6)   # top outlet
    L_right_3        = gmsh.model.geo.addLine(p6,  p7)   # wall
    L_top            = gmsh.model.geo.addLine(p7,  p8)   # wall
    L_left_1         = gmsh.model.geo.addLine(p8,  p9)   # wall above top inlet
    L_inlet_top      = gmsh.model.geo.addLine(p9,  p10)  # top inlet
    L_left_2         = gmsh.model.geo.addLine(p10, p11)  # wall between inlets
    L_inlet_bottom   = gmsh.model.geo.addLine(p11, p12)  # bottom inlet
    L_left_3         = gmsh.model.geo.addLine(p12, p1)   # wall below bottom inlet

    loop = gmsh.model.geo.addCurveLoop([
        L_bottom, L_right_1, L_outlet_bottom, L_right_2, L_outlet_top, L_right_3,
        L_top, L_left_1, L_inlet_top, L_left_2, L_inlet_bottom, L_left_3,
    ])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surf], name='Fluid')
    gmsh.model.addPhysicalGroup(1, [L_inlet_top],     name='Inlet_top')
    gmsh.model.addPhysicalGroup(1, [L_inlet_bottom],  name='Inlet_bottom')
    gmsh.model.addPhysicalGroup(1, [L_outlet_top],    name='Outlet_top')
    gmsh.model.addPhysicalGroup(1, [L_outlet_bottom], name='Outlet_bottom')
    wall_curves = [
        L_bottom, L_right_1, L_right_2, L_right_3,
        L_top, L_left_1, L_left_2, L_left_3,
    ]
    gmsh.model.addPhysicalGroup(1, wall_curves, name='Walls')

    gmsh.option.setNumber('Mesh.CharacteristicLengthMin', LC)
    gmsh.option.setNumber('Mesh.CharacteristicLengthMax', LC)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)

    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)
    gmsh.finalize()

    print(f'Wrote double pipe mesh: {output_filename}')


if __name__ == '__main__':
    create_mesh('double_pipe_2d.msh')
