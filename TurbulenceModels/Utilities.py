from dolfin import *
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET

# ----------------------------------------- #
# Utilities for all turbulence simulations  #
# ----------------------------------------- #



# ------------- Utilities for checking convergence  ------------- #

def _compute_l2_error(f1, f0):
    '''Compute l2 error of two functions: f1, f0''' 
    error = f1 - f0
    error = assemble(error**2*dx)
    return error

def _are_close(f1, f0, tol):
    '''Check if two functions: f1, f0 are sufficiently (tol) close'''
    error = _compute_l2_error(f1, f0)
    if error <= tol:
        return True, error
    else:
        return False, error

def are_close_all(fs1, fs0, tol):
    '''Check if all functions in fs1, fs0 are sufficiently (tol) close'''
    break_flag = False
    count  = 0
    errors = []

    for f1, f0 in zip(fs1, fs0):
        flag, error = _are_close(f1, f0, tol)
        errors.append(error)
        if flag == False:
            count += 1

    if count == 0:
        break_flag = True
    return  break_flag, errors

# ---------------------- Saving utilities ----------------------- #

def save_pvd_file(f, directory):
    '''Saves function f as .pvd file (to inspect in ParaView)'''
    os.makedirs(os.path.dirname(directory), exist_ok=True)
    File(directory) << f

def save_h5_file(f, directory):
    '''Saves function f as .h5 file (to load back to FEniCS)'''
    os.makedirs(os.path.dirname(directory), exist_ok=True)
    fFile = HDF5File(MPI.COMM_WORLD, directory, "w")
    fFile.write(f,"/f")
    fFile.close()

def save_list(dataset, directory):
    '''Saves python list as .txt file'''
    os.makedirs(os.path.dirname(directory), exist_ok=True)

    with open(directory, 'w+') as file:
        for value in dataset:
            file.write(str(value) + '\n')

# ---------------------- Solver utilities ----------------------- #

def calculate_cfl_time_step(u, delta_x, delta_y, relax, mesh, cfl_space=None):
    '''calculate time dt step base on cfl condition'''
    u_x = u[0]
    u_y = u[1]
    
    a = (abs(u_x) / delta_x + abs(u_y) / delta_y)

    if cfl_space is None:
        cfl_space = FunctionSpace(mesh, "DG", 0)
    local_cfl = project(1. / a, cfl_space)
    local_vals = local_cfl.vector().get_local()
    local_min = float(np.min(local_vals)) if local_vals.size else float("inf")
    global_cfl = MPI.COMM_WORLD.allreduce(local_min, op=MPI.MIN)
    return relax * global_cfl

def bound_from_bellow(f, lb):
    '''bounds function f from bellow by lb'''
    vec = f.vector()
    local_values = vec.get_local()
    np.maximum(local_values, lb, out=local_values)
    vec.set_local(local_values)
    vec.apply("insert")
    return f

def positive_part(expr):
    '''UFL positive part, matching the FluidTO wall-distance helper.'''
    zero = Constant(0.0)
    return conditional(gt(expr, zero), expr, zero)

# ------------------- Visualization utilities ------------------- #

def visualize_functions(solution_dictionary):
    '''Plots solutions'''
    num_plots = len(solution_dictionary)
    num_cols  = int(np.ceil(np.sqrt(num_plots)))  
    num_rows  = int(np.ceil(num_plots / num_cols))
    fig, _ = plt.subplots(num_rows, num_cols)

    for i, (key, f) in enumerate(solution_dictionary.items()):
        plt.subplot(num_rows, num_cols, i+1)
        if f.ufl_shape != ():
            c=plot(sqrt(dot(f, f)), title=key + ' magnitude')
        else:
            c=plot(f, title=key)
        plt.colorbar(c)
        plt.xlabel('x-direction')
        plt.ylabel('y-direction')

    fig.suptitle('Function plots')
    plt.tight_layout()
    plt.show()

def visualize_convergence(convergence_dictionary):
    '''Plots residuals'''
    num_plots = len(convergence_dictionary)
    num_cols = int(np.ceil(np.sqrt(num_plots)))  
    num_rows = int(np.ceil(num_plots / num_cols))  
    fig, axs = plt.subplots(num_rows, num_cols)
    axs = np.array(axs, ndmin=2).reshape(num_rows, num_cols)

    for i, (key, values) in enumerate(convergence_dictionary.items()):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].plot(range(1, len(values)+1), values)
        axs[row, col].set_title(key)
        axs[row, col].set_yscale("log")
        axs[row, col].set_xlabel("iterations")
        axs[row, col].set_ylabel("error (log scale)")

    fig.suptitle('Convergence plots')
    plt.tight_layout()
    plt.show()

# -------------------- Function constructor --------------------- #

def _apply_initial_condition(Space, initial_condition):
    '''Applies initial condition (Function/Constant/float)'''
    if isinstance(initial_condition, Function):
        applied_condition = initial_condition  

    else:
        if not isinstance(initial_condition, Constant):
            initial_condition = Constant(initial_condition)
        # Interpolation avoids an expensive global solve and is exact for constants.
        try:
            applied_condition = interpolate(initial_condition, Space)
        except Exception:
            # Fallback for non-interpolable inputs.
            applied_condition = project(initial_condition, Space)
    return applied_condition

def initialize_functions(Space, initial_condition=None):
    '''Initialize all functions appearing in weak form'''
    trial_f = TrialFunction(Space)
    test_f  = TestFunction(Space)
    current_f  = Function(Space)
    previous_f = Function(Space)
    
    if initial_condition:
        previous_f = _apply_initial_condition(Space, initial_condition)

    return trial_f, test_f, current_f, previous_f

def initialize_mixed_functions(Space, initial_condition=None):
    '''Initialize all functions appearing in weak form (mixed)'''    
    (trial_f1, trial_f2) = TrialFunctions(Space)
    (test_f1,   test_f2) = TestFunctions(Space)
    current_mixed  = Function(Space)
    previous_mixed = Function(Space)

    if initial_condition:
        previous_mixed = _apply_initial_condition(Space, initial_condition)

    current_f1, current_f2   = split(current_mixed)
    previous_f1, previous_f2 = split(previous_mixed)

    return trial_f1, test_f1, current_f1, previous_f1, \
           trial_f2, test_f2, current_f2, previous_f2, \
           current_mixed, previous_mixed 

# ---------------- Mesh and distance constructor ---------------- #

def _raise_xdmf_load_error(label, xdmf_path, exc):
    '''Raise a clearer mesh-loading error for common HDF5/XDMF issues.'''
    xdmf_abs_path = os.path.abspath(xdmf_path)
    sibling_h5_path = os.path.splitext(xdmf_path)[0] + '.h5'
    sibling_h5_abs_path = os.path.abspath(sibling_h5_path)
    hdf5_locking = os.environ.get('HDF5_USE_FILE_LOCKING')

    message = [
        f"Failed to load {label} from XDMF/HDF5 files.",
        f"XDMF path: {xdmf_abs_path}",
    ]

    if os.path.exists(xdmf_path):
        message.append("XDMF file exists.")
    else:
        message.append("XDMF file is missing.")

    if os.path.exists(sibling_h5_path):
        message.append(f"Sibling HDF5 path exists: {sibling_h5_abs_path}")
    else:
        message.append(
            "Sibling HDF5 path not found (the XDMF file may reference a different .h5 filename)."
        )

    message.extend([
        f"HDF5_USE_FILE_LOCKING={hdf5_locking!r}",
        "",
        "Common cause: HDF5 read failure on a Docker bind-mounted or cloud-synced folder",
        "(macOS/Windows shared folders, iCloud/Dropbox/OneDrive).",
        "",
        "Try one of these:",
        "1. Run with: HDF5_USE_FILE_LOCKING=FALSE python3 <script>.py",
        "2. Copy the mesh .xdmf/.h5 files to a container-local path (for example /tmp) and load from there.",
        "3. Move the repository to a non-synced local directory before mounting into Docker.",
        "",
        f"Original exception: {type(exc).__name__}: {exc}",
    ])

    raise RuntimeError("\n".join(message)) from exc

def _get_xdmf_attribute_name(xdmf_path):
    '''Return the first XDMF Attribute name, if present.'''
    try:
        root = ET.parse(xdmf_path).getroot()
    except ET.ParseError:
        return None

    for element in root.iter():
        if element.tag.endswith("Attribute"):
            return element.get("Name")
    return None

def _copy_xdmf_pair_to_directory(xdmf_path, destination_directory):
    '''Copy an XDMF file and its sibling .h5 file into destination_directory.'''
    xdmf_abs_path = os.path.abspath(xdmf_path)
    h5_abs_path = os.path.splitext(xdmf_abs_path)[0] + '.h5'

    staged_xdmf_path = os.path.join(destination_directory, os.path.basename(xdmf_abs_path))
    shutil.copy2(xdmf_abs_path, staged_xdmf_path)

    if os.path.exists(h5_abs_path):
        staged_h5_path = os.path.join(destination_directory, os.path.basename(h5_abs_path))
        shutil.copy2(h5_abs_path, staged_h5_path)

    return staged_xdmf_path

def _load_xdmf_with_local_copy_fallback(label, xdmf_path, load_callback):
    '''
    Load an XDMF/HDF5 pair, retrying from a temporary local copy if the original
    path fails. This avoids common Docker bind-mount and cloud-sync HDF5 issues.
    '''
    try:
        with XDMFFile(xdmf_path) as infile:
            load_callback(infile)
        return
    except Exception as direct_exc:
        try:
            with tempfile.TemporaryDirectory(prefix="fenics_xdmf_") as tmpdir:
                staged_xdmf_path = _copy_xdmf_pair_to_directory(xdmf_path, tmpdir)
                with XDMFFile(staged_xdmf_path) as infile:
                    load_callback(infile)
        except Exception as staged_exc:
            raise RuntimeError(
                "Direct XDMF/HDF5 read failed.\n"
                f"Direct read exception: {type(direct_exc).__name__}: {direct_exc}\n"
                f"Staged local-copy retry exception: {type(staged_exc).__name__}: {staged_exc}"
            ) from direct_exc

        if MPI.COMM_WORLD.Get_rank() == 0:
            print(
                f"Warning: Loaded {label} via a temporary local copy after shared-folder access "
                f"failed for {os.path.abspath(xdmf_path)}."
            )

def load_mesh_from_file(mesh_directory, facet_directory):
    '''Loads .xdmf mesh and faces mesh'''
    if not os.path.exists(mesh_directory):
        raise FileNotFoundError(f"Mesh XDMF file not found: {os.path.abspath(mesh_directory)}")
    if not os.path.exists(facet_directory):
        raise FileNotFoundError(f"Facet XDMF file not found: {os.path.abspath(facet_directory)}")

    mesh = Mesh()
    try:
        _load_xdmf_with_local_copy_fallback(
            "mesh",
            mesh_directory,
            lambda infile: infile.read(mesh),
        )
    except Exception as exc:
        _raise_xdmf_load_error("mesh", mesh_directory, exc)

    mvc = MeshValueCollection("size_t", mesh, 1)
    try:
        facet_attribute_name = _get_xdmf_attribute_name(facet_directory)
        if facet_attribute_name:
            _load_xdmf_with_local_copy_fallback(
                "facet markers",
                facet_directory,
                lambda infile: infile.read(mvc, facet_attribute_name),
            )
        else:
            _load_xdmf_with_local_copy_fallback(
                "facet markers",
                facet_directory,
                lambda infile: infile.read(mvc),
            )
    except Exception as exc:
        _raise_xdmf_load_error("facet markers", facet_directory, exc)
    marked_facets = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    return mesh, marked_facets

def _normalize_wall_markers(wall_index):
    '''Normalize wall markers to a Python set of ints.'''
    if isinstance(wall_index, (list, tuple, set, np.ndarray)):
        return {int(marker) for marker in wall_index}
    return {int(wall_index)}

def _build_wall_bcs(Space, mf, wall_index, value):
    '''Dirichlet boundary conditions on wall markers.'''
    wall_markers = sorted(_normalize_wall_markers(wall_index))
    return [DirichletBC(Space, Constant(value), mf, marker) for marker in wall_markers]

# Original Eikonal equation (for initialization) -> Yoon 2016 Eq. 16
def _calculate_eikonal_distance_field(Space, mf, wall_index, relax, custom_dx=None):
    '''Smoothened (with relaxation) Eikonal equation for wall-distance.'''
    dx_measure = custom_dx if custom_dx is not None else dx
    bcy = _build_wall_bcs(Space, mf, wall_index, 0.0)

    y = Function(Space)
    dy = TrialFunction(Space)
    z = TestFunction(Space)
    relaxation = Constant(relax)
    g = Constant(1.0)

    # Linear approximation
    F0 = inner(grad(dy), grad(z))*dx_measure - g*z*dx_measure
    a0, L0 = lhs(F0), rhs(F0)
    solve(a0==L0, y, bcy)

    # Non-linear solver
    F0  = sqrt(inner(grad(y), grad(y)) + DOLFIN_EPS)*z*dx_measure - g*z*dx_measure \
        + relaxation*inner(grad(y),grad(z))*dx_measure
    problem = NonlinearVariationalProblem(F0, y,J=derivative(F0, y), bcs=bcy)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()
    return y

# Relaxed wall equation -> Yoon 2016 Eq. 19
def calculate_relaxed_wall_distance_field_yoon_eq19(
    Space,
    mf,
    wall_index,
    relax=0.01,
    sigma_w=0.1,
    g0=20.0,
    g_floor=1.0e-12,
    newton_rtol=1.0e-8,
    newton_atol=1.0e-10,
    newton_max_iters=80,
    newton_relax=0.5,
    custom_dx=None,
):
    '''
    Yoon 2016 Eq. (19) relaxed wall equation in reciprocal-distance form G.

    Solves for G and reconstructs wall distance using Eq. (15): y = 1/G - 1/G0.
    '''
    if float(g0) <= 0.0:
        raise ValueError("Yoon Eq. (19) requires g0 > 0.")
    if float(g_floor) <= 0.0:
        raise ValueError("Yoon Eq. (19) requires g_floor > 0.")

    dx_measure = custom_dx if custom_dx is not None else dx
    sigma_w_const = Constant(float(sigma_w))
    # Robust initialization from the standard smoothed Eikonal distance.
    y_initial = _calculate_eikonal_distance_field(Space, mf, wall_index, relax, custom_dx=dx_measure)
    G = project(Constant(1.0) / (y_initial + Constant(1.0 / float(g0))), Space)
    bound_from_bellow(G, float(g_floor))

    wall_bcs = _build_wall_bcs(Space, mf, wall_index, float(g0))
    z = TestFunction(Space)

    # Weak form corresponding to Yoon 2016 Eq. (19). This is the Chapter 4
    # penalized reciprocal-distance residual with the TO penalty term set to 0:
    # |grad G|^2 + sigma_w * G * div(grad G) = (1 + 2 sigma_w) * G^4
    # using |grad G|^2 = div(G grad G) - G * div(grad G).
    F = (
        (Constant(1.0) - sigma_w_const) * inner(grad(G), grad(G)) * z * dx_measure
        - sigma_w_const * G * inner(grad(G), grad(z)) * dx_measure
        - (Constant(1.0) + Constant(2.0) * sigma_w_const) * G**4 * z * dx_measure
    )
    problem = NonlinearVariationalProblem(F, G, bcs=wall_bcs, J=derivative(F, G))
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["report"] = False
    solver.parameters["newton_solver"]["relative_tolerance"] = float(newton_rtol)
    solver.parameters["newton_solver"]["absolute_tolerance"] = float(newton_atol)
    solver.parameters["newton_solver"]["maximum_iterations"] = int(newton_max_iters)
    solver.parameters["newton_solver"]["relaxation_parameter"] = float(newton_relax)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
    solver.solve()
    bound_from_bellow(G, float(g_floor))

    return positive_part(
        Constant(1.0) / (G + Constant(float(g_floor)))
        - Constant(1.0 / float(g0))
    )

# Relaxed wall-distance Eikonal equation (Yoon 2016 Eq. 19)
def calculate_Distance_field(
    Space,
    mf,
    wall_index,
    relax=0.01,
    method='OriginalEikonal',
    sigma_w=0.1,
    g0=20.0,
    g_floor=1.0e-12,
    newton_rtol=1.0e-8,
    newton_atol=1.0e-10,
    newton_max_iters=80,
    newton_relax=0.5,
    custom_dx=None,
):
    '''computes distance to boundaries specified by wall_index on mf'''
    method_normalized = method.lower().replace('-', '_')

    if method_normalized in {'relaxedwalleikonal', 'yoon_eq19', 'relaxed_wall', 'yoon19'}:
        return calculate_relaxed_wall_distance_field_yoon_eq19(
            Space,
            mf,
            wall_index,
            relax=relax,
            sigma_w=sigma_w,
            g0=g0,
            g_floor=g_floor,
            newton_rtol=newton_rtol,
            newton_atol=newton_atol,
            newton_max_iters=newton_max_iters,
            newton_relax=newton_relax,
            custom_dx=custom_dx,
        )
    if method_normalized in {'originaleikonal', 'eikonal'}:
        return _calculate_eikonal_distance_field(Space, mf, wall_index, relax, custom_dx=custom_dx)

    raise ValueError(
        "Unknown wall-distance method '{}'. Use 'OriginalEikonal' or 'RelaxedWallEikonal'.".format(method)
    )

# --------------------------------------------------------------- #
