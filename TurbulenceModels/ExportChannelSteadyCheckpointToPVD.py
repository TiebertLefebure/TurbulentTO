from dolfin import *
from Utilities import load_h5_file, load_mesh_from_file, save_pvd_file
from Configs.ConfigChannel_SpalartAllmaras_Steady import (
    mesh_files,
    saving_directory,
    steady_sa_solver_parameters,
)


parameters["std_out_all_processes"] = False
IS_ROOT = MPI.COMM_WORLD.Get_rank() == 0

mesh, _marked_facets = load_mesh_from_file(
    mesh_files["MESH_DIRECTORY"],
    mesh_files["FACET_DIRECTORY"],
)
mesh_width = mesh.coordinates()[:, 0].max() - mesh.coordinates()[:, 0].min()


class Periodic(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)

    def map(self, x, y):
        y[0] = x[0] - mesh_width
        y[1] = x[1]


periodic = Periodic(1e-5)

V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=periodic)
Q = FunctionSpace(mesh, "CG", 1)
K = FunctionSpace(mesh, "CG", 1, constrained_domain=periodic)

fields = {
    "u": Function(V),
    "p": Function(Q),
    "nu_tilde": Function(K),
}

h5_directory = steady_sa_solver_parameters.get("RESTART_H5_DIRECTORY", saving_directory["H5_FILES"])
for key, field in fields.items():
    load_h5_file(field, "{}/{}.h5".format(h5_directory, key))
    save_pvd_file(field, "{}{}.pvd".format(saving_directory["PVD_FILES"], key))

if IS_ROOT:
    print("Exported channel checkpoint HDF5 fields from {} to {}.".format(
        h5_directory,
        saving_directory["PVD_FILES"],
    ))
