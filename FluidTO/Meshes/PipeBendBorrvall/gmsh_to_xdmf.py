import meshio
import argparse
import sys
import numpy as np
from pathlib import Path


def convert_msh_to_xdmf(msh_file, output_prefix=""):
    """
    Converts a 2D Gmsh .msh file (v2.2) with 'triangle' and 'line' cells
    into separate XDMF files for the volume mesh and facet tags.
    """
    try:
        msh = meshio.read(msh_file)
    except FileNotFoundError:
        print(f"Error: File not found at '{msh_file}'")
        sys.exit(1)

    points = msh.points
    if points.shape[1] >= 3:
        max_abs_z = float(np.max(np.abs(points[:, 2])))
        if max_abs_z > 1.0e-12:
            raise RuntimeError(
                f"Mesh appears non-planar (max |z| = {max_abs_z:.3e}); refusing XY projection."
            )
        points = points[:, :2].copy()

    cells_dict = msh.cells_dict
    cell_data_dict = msh.cell_data_dict
    physical_tags = cell_data_dict["gmsh:physical"]

    if "triangle" not in cells_dict:
        raise RuntimeError(f"No triangle cells found in {msh_file}!")

    triangles = cells_dict["triangle"]
    triangle_tags = physical_tags["triangle"]

    volume_mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        cell_data={"cell_tags": [triangle_tags]},
    )

    output_dir = Path(msh_file).resolve().parent
    volume_file = output_dir / "mesh.xdmf" if output_prefix == "" else Path(f"{output_prefix}mesh.xdmf")
    meshio.write(volume_file, volume_mesh)
    print(f"Wrote {volume_file} (triangles + cell_tags)")

    if "line" not in cells_dict:
        raise RuntimeError(f"No line cells found in {msh_file}!")

    lines = cells_dict["line"]
    line_tags = physical_tags["line"]

    facet_mesh = meshio.Mesh(
        points=points,
        cells=[("line", lines)],
        cell_data={"facet_tags": [line_tags]},
    )

    facet_file = output_dir / "facet.xdmf" if output_prefix == "" else Path(f"{output_prefix}facet.xdmf")
    meshio.write(facet_file, facet_mesh)
    print(f"Wrote {facet_file} (lines + facet_tags)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Gmsh .msh file to XDMF for FEniCS."
    )
    parser.add_argument(
        "msh_file",
        nargs="?",
        default="pipe_bend_2d.msh",
        help="Path to the input .msh file (default: pipe_bend_2d.msh)"
    )
    args = parser.parse_args()
    convert_msh_to_xdmf(args.msh_file)
