import gmsh

# ---------------------------------
# Geometry parameters
# ---------------------------------
L_X = 1.0
L_Y = 1.0

# Outlet port on right wall: centre third
OUTLET_Y_MIN = L_Y / 3.0
OUTLET_Y_MAX = 2.0 * L_Y / 3.0

# Uniform mesh size matched to N = 120 in the config
LC = L_X / 120.0


def create_mesh(output_filename='diffuser_2d.msh'):
    gmsh.initialize()
    gmsh.model.add('Diffuser_2D')

    # Corner and split points (counterclockwise from bottom-left)
    p1 = gmsh.model.geo.addPoint(0.0,        0.0,          0.0, LC)  # bottom-left
    p2 = gmsh.model.geo.addPoint(L_X,        0.0,          0.0, LC)  # bottom-right
    p3 = gmsh.model.geo.addPoint(L_X,        OUTLET_Y_MIN, 0.0, LC)  # outlet bottom
    p4 = gmsh.model.geo.addPoint(L_X,        OUTLET_Y_MAX, 0.0, LC)  # outlet top
    p5 = gmsh.model.geo.addPoint(L_X,        L_Y,          0.0, LC)  # top-right
    p6 = gmsh.model.geo.addPoint(0.0,        L_Y,          0.0, LC)  # top-left

    L_bottom       = gmsh.model.geo.addLine(p1, p2)  # wall
    L_right_bottom = gmsh.model.geo.addLine(p2, p3)  # wall
    L_outlet       = gmsh.model.geo.addLine(p3, p4)  # outlet
    L_right_top    = gmsh.model.geo.addLine(p4, p5)  # wall
    L_top          = gmsh.model.geo.addLine(p5, p6)  # wall
    L_inlet        = gmsh.model.geo.addLine(p6, p1)  # inlet (full left wall)

    loop = gmsh.model.geo.addCurveLoop([
        L_bottom, L_right_bottom, L_outlet, L_right_top, L_top, L_inlet,
    ])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surf], name='Fluid')
    gmsh.model.addPhysicalGroup(1, [L_inlet], name='Inlet')
    gmsh.model.addPhysicalGroup(1, [L_outlet], name='Outlet')
    wall_curves = [L_bottom, L_right_bottom, L_right_top, L_top]
    gmsh.model.addPhysicalGroup(1, wall_curves, name='Walls')

    gmsh.option.setNumber('Mesh.CharacteristicLengthMin', LC)
    gmsh.option.setNumber('Mesh.CharacteristicLengthMax', LC)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)

    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)
    gmsh.finalize()

    print(f'Wrote diffuser mesh: {output_filename}')


if __name__ == '__main__':
    create_mesh('diffuser_2d.msh')
