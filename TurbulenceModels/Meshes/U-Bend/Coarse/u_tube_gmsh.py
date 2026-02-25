import gmsh
import math

# -----------------------------
# Geometry parameters (meters)
# -----------------------------
MM = 1e-3

R_PIPE = 14.0 * MM      # Pipe radius
R_CURV = 125.0 * MM     # Centerline radius of bend

# Inner/outer radii of the bend walls
R_INNER = R_CURV - R_PIPE
R_OUTER = R_CURV + R_PIPE

D_PIPE = 2.0 * R_PIPE        # Pipe diameter
H_LEG = 30.0 * D_PIPE        # Straight-leg length (both sides) = 30D = 840 mm

# Coarse characteristic element size
LC = D_PIPE / 10.0           # Adjust for coarser/finer global mesh


def create_mesh(output_filename="u_bend_2d.msh"):
    """Generates a coarse 2D mesh of a U-bend pipe using Gmsh."""
    # -----------------------------
    # Initialize Gmsh
    # -----------------------------
    gmsh.initialize()
    gmsh.model.add("U_bend_2D")

    # Center of curvature at origin
    cx, cy = 0.0, 0.0

    # Convenience function for points
    def add_point(x, y):
        return gmsh.model.geo.addPoint(x, y, 0.0, LC)

    # -----------------------------
    # Points
    # -----------------------------
    # Bottom points (theta = 0 for right, pi for left)
    p_out_R_bot = add_point(cx + R_OUTER, cy)    # outer right bottom
    p_out_L_bot = add_point(cx - R_OUTER, cy)    # outer left bottom
    p_in_R_bot  = add_point(cx + R_INNER, cy)    # inner right bottom
    p_in_L_bot  = add_point(cx - R_INNER, cy)    # inner left bottom

    # Mid points to enforce bottom semicircle
    p_out_mid   = add_point(cx, cy - R_OUTER)
    p_in_mid    = add_point(cx, cy - R_INNER)

    # Top of legs (shift up by H_leg)
    p_out_R_top = add_point(cx + R_OUTER, cy + H_LEG)
    p_in_R_top  = add_point(cx + R_INNER, cy + H_LEG)
    p_out_L_top = add_point(cx - R_OUTER, cy + H_LEG)
    p_in_L_top  = add_point(cx - R_INNER, cy + H_LEG)

    # Center for circular arcs
    p_center = add_point(cx, cy)

    # -----------------------------
    # Lines and arcs
    # -----------------------------
    # Outer walls (legs)
    L_out_left  = gmsh.model.geo.addLine(p_out_L_top, p_out_L_bot)
    L_out_right = gmsh.model.geo.addLine(p_out_R_bot, p_out_R_top)

    # Inner walls (legs)
    L_in_right  = gmsh.model.geo.addLine(p_in_R_top, p_in_R_bot)
    L_in_left   = gmsh.model.geo.addLine(p_in_L_bot, p_in_L_top)

    # Inlet and outlet (top cross-sections)
    L_inlet  = gmsh.model.geo.addLine(p_in_L_top,  p_out_L_top)   # inlet (left, top)
    L_outlet = gmsh.model.geo.addLine(p_out_R_top, p_in_R_top)    # outlet (right, top)

    # Outer bend: left bottom -> mid bottom -> right bottom (bottom semicircle)
    A_out_1 = gmsh.model.geo.addCircleArc(p_out_L_bot, p_center, p_out_mid)
    A_out_2 = gmsh.model.geo.addCircleArc(p_out_mid,   p_center, p_out_R_bot)

    # Inner bend: right bottom -> mid bottom -> left bottom (bottom semicircle)
    A_in_1  = gmsh.model.geo.addCircleArc(p_in_R_bot,  p_center, p_in_mid)
    A_in_2  = gmsh.model.geo.addCircleArc(p_in_mid,    p_center, p_in_L_bot)

    # -----------------------------
    # Curve loop and surface
    # -----------------------------
    # Order matters: follow boundary around the fluid region
    loop = gmsh.model.geo.addCurveLoop([
        L_inlet,        # inlet (top left, inner -> outer)
        L_out_left,     # outer left leg down
        A_out_1,        # outer bend left -> mid
        A_out_2,        # outer bend mid -> right
        L_out_right,    # outer right leg up
        L_outlet,       # outlet (top right, outer -> inner)
        L_in_right,     # inner right leg down
        A_in_1,         # inner bend right -> mid
        A_in_2,         # inner bend mid -> left
        L_in_left       # inner left leg up
    ])

    surf = gmsh.model.geo.addPlaneSurface([loop])

    # Finalize CAD -> Gmsh model
    gmsh.model.geo.synchronize()

    # -----------------------------
    # Physical groups (for FEniCS)
    # -----------------------------
    # 2D region: fluid
    gmsh.model.addPhysicalGroup(2, [surf], name="Fluid")

    # 1D boundaries
    gmsh.model.addPhysicalGroup(1, [L_inlet],  name="Inlet")
    gmsh.model.addPhysicalGroup(1, [L_outlet], name="Outlet")
    gmsh.model.addPhysicalGroup(
        1,
        [L_out_left, L_out_right, L_in_left, L_in_right,
         A_out_1, A_out_2, A_in_1, A_in_2],
        name="Walls"
    )

    # -----------------------------
    # Mesh + export
    # -----------------------------
    # Use MSH v2.2 for compatibility with meshio/FEniCS
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    # (Optional) pick a simple 2D algorithm
    # gmsh.option.setNumber("Mesh.Algorithm", 6)  # 6 = Frontal-Delaunay (2D)

    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)

    gmsh.finalize()

    print(f"Wrote coarse mesh: {output_filename}")

if __name__ == "__main__":
    create_mesh("u_bend_2d.msh")


### (1) u_tube_gmsh.py ###

### INPUT ### 

# cd /Users/tiebertlefebure/Documents/FEniCS-Tiebert/Turbulence_models/Meshes/PipeBend/Coarse
# python3 u_tube_gmsh.py

### OUTPUT ###

# "Wrote coarse mesh: u_bend_2d.msh"
