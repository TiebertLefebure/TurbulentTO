import gmsh

# ---------------------------------
# Geometry parameters from Dilgen 2018 Fig. 15
# ---------------------------------
H = 0.1

DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = 10.0 * H
DESIGN_Y_MAX = 10.0 * H

INLET_BLOCK_X_MIN = -2.0 * H
INLET_BLOCK_X_MAX = DESIGN_X_MIN
INLET_BLOCK_Y_MIN = 6.0 * H
INLET_BLOCK_Y_MAX = 10.0 * H
INLET_Y_MIN = 7.0 * H
INLET_Y_MAX = 9.0 * H

TOP_BLOCK_X_MIN = 6.0 * H
TOP_BLOCK_X_MAX = 10.0 * H
TOP_BLOCK_Y_MIN = DESIGN_Y_MAX
TOP_BLOCK_Y_MAX = 20.0 * H
TOP_OUTLET_X_MIN = 7.0 * H
TOP_OUTLET_X_MAX = 9.0 * H

RIGHT_BLOCK_X_MIN = DESIGN_X_MAX
RIGHT_BLOCK_X_MAX = 20.0 * H
RIGHT_BLOCK_Y_MIN = 4.0 * H
RIGHT_BLOCK_Y_MAX = 8.0 * H
RIGHT_OUTLET_Y_MIN = 5.0 * H
RIGHT_OUTLET_Y_MAX = 7.0 * H

BOTTOM_BLOCK_X_MIN = 0.0 * H
BOTTOM_BLOCK_X_MAX = 4.0 * H
BOTTOM_BLOCK_Y_MIN = -10.0 * H
BOTTOM_BLOCK_Y_MAX = DESIGN_Y_MIN
BOTTOM_OUTLET_X_MIN = 1.0 * H
BOTTOM_OUTLET_X_MAX = 3.0 * H

# Uniform mesh size matched to the design-square reference resolution.
LC = 0.01


def create_mesh(output_filename="manifold_dilgen_2d.msh"):
    gmsh.initialize()
    gmsh.model.add("ManifoldDilgen_2D")

    # Outer boundary points traced counterclockwise.
    p1 = gmsh.model.geo.addPoint(BOTTOM_BLOCK_X_MIN, BOTTOM_BLOCK_Y_MIN, 0.0, LC)
    p2 = gmsh.model.geo.addPoint(BOTTOM_OUTLET_X_MIN, BOTTOM_BLOCK_Y_MIN, 0.0, LC)
    p3 = gmsh.model.geo.addPoint(BOTTOM_OUTLET_X_MAX, BOTTOM_BLOCK_Y_MIN, 0.0, LC)
    p4 = gmsh.model.geo.addPoint(BOTTOM_BLOCK_X_MAX, BOTTOM_BLOCK_Y_MIN, 0.0, LC)
    p5 = gmsh.model.geo.addPoint(BOTTOM_BLOCK_X_MAX, DESIGN_Y_MIN, 0.0, LC)
    p6 = gmsh.model.geo.addPoint(DESIGN_X_MAX, DESIGN_Y_MIN, 0.0, LC)
    p7 = gmsh.model.geo.addPoint(DESIGN_X_MAX, RIGHT_BLOCK_Y_MIN, 0.0, LC)
    p8 = gmsh.model.geo.addPoint(RIGHT_BLOCK_X_MAX, RIGHT_BLOCK_Y_MIN, 0.0, LC)
    p9 = gmsh.model.geo.addPoint(RIGHT_BLOCK_X_MAX, RIGHT_OUTLET_Y_MIN, 0.0, LC)
    p10 = gmsh.model.geo.addPoint(RIGHT_BLOCK_X_MAX, RIGHT_OUTLET_Y_MAX, 0.0, LC)
    p11 = gmsh.model.geo.addPoint(RIGHT_BLOCK_X_MAX, RIGHT_BLOCK_Y_MAX, 0.0, LC)
    p12 = gmsh.model.geo.addPoint(DESIGN_X_MAX, RIGHT_BLOCK_Y_MAX, 0.0, LC)
    p13 = gmsh.model.geo.addPoint(TOP_BLOCK_X_MAX, TOP_BLOCK_Y_MAX, 0.0, LC)
    p14 = gmsh.model.geo.addPoint(TOP_OUTLET_X_MAX, TOP_BLOCK_Y_MAX, 0.0, LC)
    p15 = gmsh.model.geo.addPoint(TOP_OUTLET_X_MIN, TOP_BLOCK_Y_MAX, 0.0, LC)
    p16 = gmsh.model.geo.addPoint(TOP_BLOCK_X_MIN, TOP_BLOCK_Y_MAX, 0.0, LC)
    p17 = gmsh.model.geo.addPoint(TOP_BLOCK_X_MIN, DESIGN_Y_MAX, 0.0, LC)
    p18 = gmsh.model.geo.addPoint(DESIGN_X_MIN, DESIGN_Y_MAX, 0.0, LC)
    p19 = gmsh.model.geo.addPoint(INLET_BLOCK_X_MIN, INLET_BLOCK_Y_MAX, 0.0, LC)
    p20 = gmsh.model.geo.addPoint(INLET_BLOCK_X_MIN, INLET_Y_MAX, 0.0, LC)
    p21 = gmsh.model.geo.addPoint(INLET_BLOCK_X_MIN, INLET_Y_MIN, 0.0, LC)
    p22 = gmsh.model.geo.addPoint(INLET_BLOCK_X_MIN, INLET_BLOCK_Y_MIN, 0.0, LC)
    p23 = gmsh.model.geo.addPoint(DESIGN_X_MIN, INLET_BLOCK_Y_MIN, 0.0, LC)
    p24 = gmsh.model.geo.addPoint(DESIGN_X_MIN, DESIGN_Y_MIN, 0.0, LC)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)    # bottom outlet
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p9)
    l9 = gmsh.model.geo.addLine(p9, p10)   # right outlet
    l10 = gmsh.model.geo.addLine(p10, p11)
    l11 = gmsh.model.geo.addLine(p11, p12)
    l12 = gmsh.model.geo.addLine(p12, p13)
    l13 = gmsh.model.geo.addLine(p13, p14)
    l14 = gmsh.model.geo.addLine(p14, p15) # top outlet
    l15 = gmsh.model.geo.addLine(p15, p16)
    l16 = gmsh.model.geo.addLine(p16, p17)
    l17 = gmsh.model.geo.addLine(p17, p18)
    l18 = gmsh.model.geo.addLine(p18, p19)
    l19 = gmsh.model.geo.addLine(p19, p20)
    l20 = gmsh.model.geo.addLine(p20, p21) # inlet
    l21 = gmsh.model.geo.addLine(p21, p22)
    l22 = gmsh.model.geo.addLine(p22, p23)
    l23 = gmsh.model.geo.addLine(p23, p24)
    l24 = gmsh.model.geo.addLine(p24, p1)

    loop = gmsh.model.geo.addCurveLoop([
        l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12,
        l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24,
    ])
    surface = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surface], name="Fluid")
    gmsh.model.addPhysicalGroup(1, [l20], name="Inlet")
    gmsh.model.addPhysicalGroup(1, [l14], name="Outlet_top")
    gmsh.model.addPhysicalGroup(1, [l9], name="Outlet_right")
    gmsh.model.addPhysicalGroup(1, [l2], name="Outlet_bottom")
    wall_curves = [l1, l3, l4, l5, l6, l7, l8, l10, l11, l12, l13, l15, l16, l17, l18, l19, l21, l22, l23, l24]
    gmsh.model.addPhysicalGroup(1, wall_curves, name="Walls")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", LC)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)
    gmsh.finalize()

    print(f"Wrote manifold mesh: {output_filename}")


if __name__ == "__main__":
    create_mesh("manifold_dilgen_2d.msh")
