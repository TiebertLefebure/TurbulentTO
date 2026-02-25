import meshio
import argparse
import sys

def convert_msh_to_xdmf(msh_file, output_prefix=""):
    """
    Converts a 2D Gmsh .msh file (v2.2) with 'triangle' and 'line' cells
    into separate XDMF files for the volume mesh and facet tags.
    """
    # -----------------------------------------
    # Read the Gmsh .msh file
    # -----------------------------------------
    try:
        msh = meshio.read(msh_file)
    except FileNotFoundError:
        print(f"Error: File not found at '{msh_file}'")
        sys.exit(1)

    # Point coordinates
    points = msh.points

    # Connectivity of cells (1D lines, 2D triangles, etc.)
    cells_dict = msh.cells_dict

    # Gmsh physical tags are stored under "gmsh:physical"
    cell_data_dict = msh.cell_data_dict
    physical_tags = cell_data_dict["gmsh:physical"]

    # -----------------------------------------
    # 2D domain mesh (triangles) -> mesh.xdmf
    # -----------------------------------------
    if "triangle" not in cells_dict:
        raise RuntimeError(f"No triangle cells found in {msh_file}!")

    triangles = cells_dict["triangle"]
    triangle_tags = physical_tags["triangle"]

    volume_mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        cell_data={"cell_tags": [triangle_tags]},
    )

    volume_file = f"{output_prefix}mesh.xdmf"
    meshio.write(volume_file, volume_mesh)
    print(f"Wrote {volume_file} (triangles + cell_tags)")

    # -----------------------------------------
    # 1D boundary mesh (lines) -> facet.xdmf
    # -----------------------------------------
    if "line" not in cells_dict:
        raise RuntimeError(f"No line cells found in {msh_file}!")

    lines = cells_dict["line"]
    line_tags = physical_tags["line"]

    facet_mesh = meshio.Mesh(
        points=points,
        cells=[("line", lines)],
        cell_data={"facet_tags": [line_tags]},
    )

    facet_file = f"{output_prefix}facet.xdmf"
    meshio.write(facet_file, facet_mesh)
    print(f"Wrote {facet_file} (lines + facet_tags)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Gmsh .msh file to XDMF for FEniCS."
    )
    parser.add_argument(
        "msh_file",
        nargs="?",
        default="u_bend_2d.msh",
        help="Path to the input .msh file (default: u_bend_2d.msh)"
    )
    args = parser.parse_args()
    convert_msh_to_xdmf(args.msh_file)


# 3) gmsh_to_xdmf.py

# INPUT

# python3 gmsh_to_xdmf.py

# OUTPUT

# mesh.xdmf
# mesh.h5
# facet.xdmf
# facet.h5

# "Wrote mesh.xdmf (triangles + cell_tags)"
# "Wrote facet.xdmf (lines + facet_tags)"
