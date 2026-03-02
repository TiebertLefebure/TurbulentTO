from dolfin import (
    Mesh,
    MeshValueCollection,
    MeshFunction,
    XDMFFile,
)
import numpy as np

def test_xdmf_mesh(mesh_file="mesh.xdmf", facet_file="facet.xdmf"):
    """
    Reads and tests a FEniCS mesh and its associated facet tags from XDMF files.
    """
    # -----------------------------
    # Read volume mesh (cells)
    # -----------------------------
    mesh = Mesh()
    with XDMFFile(mesh_file) as xdmf:
        xdmf.read(mesh)

    dim = mesh.topology().dim()
    print(f"--- Reading {mesh_file} ---")
    print("Mesh dimension:", dim)
    print("Number of cells:", mesh.num_cells())
    print("Number of vertices:", mesh.num_vertices())

    # -----------------------------
    # Read cell tags ("Fluid", etc.)
    # -----------------------------
    mvc_cells = MeshValueCollection("size_t", mesh, dim)
    with XDMFFile(mesh_file) as xdmf:
        xdmf.read(mvc_cells, "cell_tags")

    cell_tags = MeshFunction("size_t", mesh, mvc_cells)

    cell_values = np.unique(cell_tags.array())
    print("Unique cell tags:", cell_values)

    # -----------------------------
    # Read facet tags ("Inlet", "Outlet", "Walls")
    # -----------------------------
    print(f"\n--- Reading {facet_file} ---")
    mvc_facets = MeshValueCollection("size_t", mesh, dim - 1)
    with XDMFFile(facet_file) as xdmf:
        xdmf.read(mvc_facets, "facet_tags")

    facet_tags = MeshFunction("size_t", mesh, mvc_facets)

    facet_values = np.unique(facet_tags.array())
    print("Unique facet tags:", facet_values)

    print("\nDone reading mesh and tags in FEniCS.")

if __name__ == "__main__":
    test_xdmf_mesh()




# 4) test_mesh.py

# INPUT

#docker run -ti --rm \
#  -v "$(pwd)":/home/fenics/shared \
#  -w /home/fenics/shared \
#  quay.io/fenicsproject/stable:current 

# python3 test_mesh.py



# OUTPUT

#--- Reading mesh.xdmf ---
#Mesh dimension: 2
#Number of cells: 8228
#Number of vertices: 4466
#Unique cell tags: [1]

#--- Reading facet.xdmf ---
#Unique facet tags: [2 3 4]

#Done reading mesh and tags in FEniCS.

