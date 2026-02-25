import meshio
import argparse
import sys

def inspect_mesh_tags(msh_file):
    """Reads a .msh file and prints its physical group information."""
    try:
        msh = meshio.read(msh_file)
    except FileNotFoundError:
        print(f"Error: File not found at '{msh_file}'")
        sys.exit(1)

    print(f"Inspecting physical groups in: {msh_file}")
    print("Field data (name -> [tag, dim]):")
    if not msh.field_data:
        print("  No physical groups found in the mesh.")
        return

    for name, data in msh.field_data.items():
        tag, dim = data
        print(f"  {name}: tag = {tag}, dim = {dim}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect physical group tags in a Gmsh .msh file."
    )
    parser.add_argument(
        "msh_file",
        nargs="?",
        default="u_bend_2d.msh",
        help="Path to the .msh file (default: u_bend_2d.msh)"
    )
    args = parser.parse_args()
    inspect_mesh_tags(args.msh_file)

    ### (2) inspect_tags.py ###

    ### INPUT ###

    # python3 inspect_tags.py

    ### OUTPUT ###