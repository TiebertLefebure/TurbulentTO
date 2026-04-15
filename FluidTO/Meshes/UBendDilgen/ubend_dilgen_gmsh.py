import gmsh

# ---------------------------------
# Geometry parameters from Dilgen 2018 Fig. 7
# ---------------------------------
H = 0.1

# The optimization/design domain is the 10H x 10H square on the right.
DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = 10.0 * H
DESIGN_Y_MAX = 10.0 * H

# Passive inlet / outlet block on the left of the design square.
LEFT_BLOCK_X_MIN = -2.0 * H
LEFT_BLOCK_X_MAX = DESIGN_X_MIN

CORNER_SOLID_HEIGHT = 2.5 * H
PORT_HEIGHT = 2.0 * H
BAFFLE_THICKNESS = 1.0 * H

OUTLET_Y_MIN = CORNER_SOLID_HEIGHT
OUTLET_Y_MAX = OUTLET_Y_MIN + PORT_HEIGHT
BAFFLE_Y_MIN = OUTLET_Y_MAX
BAFFLE_Y_MAX = BAFFLE_Y_MIN + BAFFLE_THICKNESS
INLET_Y_MIN = BAFFLE_Y_MAX
INLET_Y_MAX = INLET_Y_MIN + PORT_HEIGHT

# Uniform mesh size matched to N = 120 across the 10H design square.
LC = (DESIGN_X_MAX - DESIGN_X_MIN) / 120.0


def create_mesh(output_filename="ubend_dilgen_2d.msh"):
    gmsh.initialize()
    gmsh.model.add("UBendDilgen_2D")

    p1 = gmsh.model.geo.addPoint(LEFT_BLOCK_X_MIN, DESIGN_Y_MIN, 0.0, LC)
    p2 = gmsh.model.geo.addPoint(DESIGN_X_MAX, DESIGN_Y_MIN, 0.0, LC)
    p3 = gmsh.model.geo.addPoint(DESIGN_X_MAX, DESIGN_Y_MAX, 0.0, LC)
    p4 = gmsh.model.geo.addPoint(LEFT_BLOCK_X_MIN, DESIGN_Y_MAX, 0.0, LC)
    p5 = gmsh.model.geo.addPoint(LEFT_BLOCK_X_MIN, INLET_Y_MAX, 0.0, LC)
    p6 = gmsh.model.geo.addPoint(LEFT_BLOCK_X_MIN, INLET_Y_MIN, 0.0, LC)
    p7 = gmsh.model.geo.addPoint(LEFT_BLOCK_X_MIN, OUTLET_Y_MAX, 0.0, LC)
    p8 = gmsh.model.geo.addPoint(LEFT_BLOCK_X_MIN, OUTLET_Y_MIN, 0.0, LC)

    l_bottom = gmsh.model.geo.addLine(p1, p2)
    l_right = gmsh.model.geo.addLine(p2, p3)
    l_top = gmsh.model.geo.addLine(p3, p4)
    l_left_top = gmsh.model.geo.addLine(p4, p5)
    l_inlet = gmsh.model.geo.addLine(p5, p6)
    l_left_mid = gmsh.model.geo.addLine(p6, p7)
    l_outlet = gmsh.model.geo.addLine(p7, p8)
    l_left_bottom = gmsh.model.geo.addLine(p8, p1)

    loop = gmsh.model.geo.addCurveLoop(
        [l_bottom, l_right, l_top, l_left_top, l_inlet, l_left_mid, l_outlet, l_left_bottom]
    )
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surf], name="Fluid")
    gmsh.model.addPhysicalGroup(1, [l_inlet], name="Inlet")
    gmsh.model.addPhysicalGroup(1, [l_outlet], name="Outlet")
    gmsh.model.addPhysicalGroup(1, [l_bottom, l_right, l_top, l_left_top, l_left_mid, l_left_bottom], name="Walls")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", LC)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)
    gmsh.finalize()

    print(f"Wrote U-bend Dilgen mesh: {output_filename}")


if __name__ == "__main__":
    create_mesh("ubend_dilgen_2d.msh")
