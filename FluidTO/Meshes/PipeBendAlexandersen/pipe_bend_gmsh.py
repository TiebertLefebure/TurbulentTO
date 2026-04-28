import gmsh


# ---------------------------------
# Geometry parameters
# ---------------------------------
L_X = 1.0
L_Y = 1.0

# Alexandersen pipe-bend port layout.
INLET_WIDTH = 0.2
INLET_TOP_OFFSET = 0.1
OUTLET_WIDTH = 0.2
OUTLET_RIGHT_OFFSET = 0.1

INLET_Y_MAX = L_Y - INLET_TOP_OFFSET          # 0.9
INLET_Y_MIN = INLET_Y_MAX - INLET_WIDTH       # 0.7
OUTLET_X_MAX = L_X - OUTLET_RIGHT_OFFSET      # 0.9
OUTLET_X_MIN = OUTLET_X_MAX - OUTLET_WIDTH    # 0.7

# Uniform mesh size matched to N = 120 in the config.
LC = L_X / 120.0


def create_mesh(output_filename="pipe_bend_2d.msh"):
    gmsh.initialize()
    gmsh.model.add("PipeBendAlexandersen_Laminar_2D")

    # Points counterclockwise from bottom-left.
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, LC)
    p2 = gmsh.model.geo.addPoint(OUTLET_X_MIN, 0.0, 0.0, LC)
    p3 = gmsh.model.geo.addPoint(OUTLET_X_MAX, 0.0, 0.0, LC)
    p4 = gmsh.model.geo.addPoint(L_X, 0.0, 0.0, LC)
    p5 = gmsh.model.geo.addPoint(L_X, L_Y, 0.0, LC)
    p6 = gmsh.model.geo.addPoint(0.0, L_Y, 0.0, LC)
    p7 = gmsh.model.geo.addPoint(0.0, INLET_Y_MAX, 0.0, LC)
    p8 = gmsh.model.geo.addPoint(0.0, INLET_Y_MIN, 0.0, LC)

    l_bot_left = gmsh.model.geo.addLine(p1, p2)
    l_outlet = gmsh.model.geo.addLine(p2, p3)
    l_bot_right = gmsh.model.geo.addLine(p3, p4)
    l_right = gmsh.model.geo.addLine(p4, p5)
    l_top = gmsh.model.geo.addLine(p5, p6)
    l_left_top = gmsh.model.geo.addLine(p6, p7)
    l_inlet = gmsh.model.geo.addLine(p7, p8)
    l_left_bot = gmsh.model.geo.addLine(p8, p1)

    loop = gmsh.model.geo.addCurveLoop([
        l_bot_left, l_outlet, l_bot_right, l_right,
        l_top, l_left_top, l_inlet, l_left_bot,
    ])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surf], name="Fluid")
    gmsh.model.addPhysicalGroup(1, [l_inlet], name="Inlet")
    gmsh.model.addPhysicalGroup(1, [l_outlet], name="Outlet")
    wall_curves = [l_bot_left, l_bot_right, l_right, l_top, l_left_top, l_left_bot]
    gmsh.model.addPhysicalGroup(1, wall_curves, name="Walls")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", LC)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)
    gmsh.finalize()

    print(f"Wrote pipe bend mesh: {output_filename}")


if __name__ == "__main__":
    create_mesh("pipe_bend_2d.msh")
