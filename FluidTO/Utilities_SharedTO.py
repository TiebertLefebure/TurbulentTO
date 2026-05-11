import argparse
import glob
import importlib
import importlib.util
import os
import re
import shutil
import sys
import tempfile
from time import localtime, strftime

from dolfin import (
    Constant,
    File,
    Function,
    MPI,
    Mesh,
    MeshFunction,
    MeshValueCollection,
    SubDomain,
    XDMFFile,
    assemble,
    avg,
    cells,
    cpp,
    near,
)


def _resolve_config_path_candidates(raw_config_arg):
    normalized_arg = raw_config_arg.strip()
    candidates = []

    def _add_candidate(path_value):
        if not path_value:
            return
        abs_path = path_value if os.path.isabs(path_value) else os.path.abspath(path_value)
        if abs_path not in candidates:
            candidates.append(abs_path)

    if normalized_arg.endswith(".py"):
        _add_candidate(normalized_arg)
    else:
        _add_candidate(normalized_arg + ".py")

    path_like_arg = normalized_arg.replace("\\", os.sep).replace("/", os.sep)
    if path_like_arg.endswith(".py"):
        _add_candidate(path_like_arg)
    else:
        _add_candidate(path_like_arg + ".py")

    module_like_arg = normalized_arg[:-3] if normalized_arg.endswith(".py") else normalized_arg
    _add_candidate(module_like_arg.replace(".", os.sep) + ".py")
    return candidates


def _load_config_module_from_path(config_path):
    module_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create an import spec for config {}.".format(config_path))

    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(module_name, None)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module_name, module


def load_config_module_from_cli():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        required=True,
        help="Python module name or repo-relative path for solver configuration (required).",
    )
    args, _unknown = parser.parse_known_args()
    config_arg = args.config.strip()
    if not config_arg:
        raise ValueError("Empty config module name is not allowed.")

    path_like = config_arg.endswith(".py") or (os.sep in config_arg) or ("/" in config_arg) or ("\\" in config_arg)
    if path_like:
        for candidate_path in _resolve_config_path_candidates(config_arg):
            if os.path.isfile(candidate_path):
                return _load_config_module_from_path(candidate_path)
        raise FileNotFoundError("Could not find config file for '{}'.".format(config_arg))

    module_name = config_arg[:-3] if config_arg.endswith(".py") else config_arg
    module_name = module_name.replace("\\", ".").replace("/", ".").strip(".")
    if not module_name:
        raise ValueError("Empty config module name is not allowed.")
    return module_name, importlib.import_module(module_name)


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _format_numeric_values(values):
    return ", ".join("{:.4g}".format(value) for value in values)


def _expand_numeric_config_value(config_values, names, count, default=None, required=False):
    for name in names:
        if name in config_values:
            raw_value = config_values[name]
            break
    else:
        if default is None:
            if required:
                raise ValueError("Expected one of {} in the configuration.".format(", ".join(names)))
            return None
        raw_value = default

    values = [float(value) for value in as_list(raw_value)]
    if len(values) == 1 and count > 1:
        values *= count
    if len(values) != count:
        raise ValueError(
            "{} must provide either one value or {} values, got {}.".format(
                names[0], count, len(values),
            )
        )
    return values


def _first_config_value(config_values, names):
    for name in names:
        if name in config_values:
            return config_values[name]
    return None


def eddy_viscosity_ratio_from_turbulence_intensity(
    intensity,
    nu_lam,
    reference_velocity=None,
    reference_length=None,
    length_scale=None,
    length_scale_ratio=None,
    reynolds_number=None,
    c_mu=0.09,
):
    """Estimate nu_t / nu_lam from inlet turbulence intensity and length scale.

    This uses the standard k-epsilon inlet estimate:
        k = 3/2 * (U * I)^2
        epsilon = C_mu^(3/4) * k^(3/2) / ell
        nu_t = C_mu * k^2 / epsilon

    which gives:
        nu_t / nu = C_mu^(1/4) * sqrt(3/2) * I * U * ell / nu

    If reynolds_number is supplied, length_scale_ratio is interpreted as ell/L_ref,
    where ell is the inlet turbulence length scale.
    Otherwise reference_velocity and length_scale are used directly.
    """
    intensity = float(intensity)
    nu_lam = float(nu_lam)
    if intensity <= 0.0:
        raise ValueError("SA_TURBULENCE_INTENSITY must be positive.")
    if nu_lam <= 0.0:
        raise ValueError("nu_lam must be positive.")

    factor = (float(c_mu) ** 0.25) * (1.5 ** 0.5)

    if reynolds_number is not None:
        reynolds_number = abs(float(reynolds_number))
        if reynolds_number <= 0.0:
            raise ValueError("SA_REYNOLDS_NUMBER must be positive.")
        if length_scale_ratio is None:
            if length_scale is None or reference_length is None:
                raise ValueError(
                    "Using SA_REYNOLDS_NUMBER requires SA_TURBULENCE_LENGTH_SCALE_RATIO "
                    "or both SA_TURBULENCE_LENGTH_SCALE and SA_REFERENCE_LENGTH."
                )
            length_scale_ratio = float(length_scale) / float(reference_length)
        return factor * intensity * reynolds_number * float(length_scale_ratio)

    if length_scale is None:
        if length_scale_ratio is None or reference_length is None:
            raise ValueError(
                "SA_TURBULENCE_INTENSITY requires either SA_TURBULENCE_LENGTH_SCALE "
                "or SA_TURBULENCE_LENGTH_SCALE_RATIO with a reference length."
            )
        length_scale = float(length_scale_ratio) * float(reference_length)

    if reference_velocity is None:
        raise ValueError(
            "SA_TURBULENCE_INTENSITY requires SA_REFERENCE_VELOCITY, U_BULK_INLET, "
            "U_MAX_INLET, or SA_REYNOLDS_NUMBER."
        )
    return factor * intensity * abs(float(reference_velocity)) * float(length_scale) / nu_lam


def build_sa_inlet_nu_tilde_targets(config_values, inlet_count, nu_lam, nu_tilde_from_ratio):
    """Build inlet nu_tilde values from the configured SA inlet turbulence model."""
    if "SA_MUT_RATIOS" in config_values or "SA_MUT_RATIO" in config_values:
        ratio_targets = _expand_numeric_config_value(
            config_values,
            ("SA_MUT_RATIOS", "SA_MUT_RATIO"),
            inlet_count,
            required=True,
        )
        nu_tilde_targets = [nu_tilde_from_ratio(ratio, nu_lam) for ratio in ratio_targets]
        description = "eddy-viscosity ratio nu_t/nu_lam = {}".format(
            _format_numeric_values(ratio_targets)
        )
        return nu_tilde_targets, description

    uses_intensity = (
        "SA_TURBULENCE_INTENSITIES" in config_values
        or "SA_TURBULENCE_INTENSITY" in config_values
    )
    if uses_intensity:
        intensities = _expand_numeric_config_value(
            config_values,
            ("SA_TURBULENCE_INTENSITIES", "SA_TURBULENCE_INTENSITY"),
            inlet_count,
            required=True,
        )
        length_scales = _expand_numeric_config_value(
            config_values,
            ("SA_TURBULENCE_LENGTH_SCALES", "SA_TURBULENCE_LENGTH_SCALE"),
            inlet_count,
        )
        length_scale_ratios = None
        if length_scales is None:
            length_scale_ratios = _expand_numeric_config_value(
                config_values,
                ("SA_TURBULENCE_LENGTH_SCALE_RATIOS", "SA_TURBULENCE_LENGTH_SCALE_RATIO"),
                inlet_count,
                default=0.07,
            )

        reynolds_numbers = _expand_numeric_config_value(
            config_values,
            ("SA_REYNOLDS_NUMBERS", "SA_REYNOLDS_NUMBER", "REYNOLDS_NUMBERS", "REYNOLDS_NUMBER", "RE"),
            inlet_count,
        )
        reference_velocity_default = _first_config_value(
            config_values,
            ("U_BULK_INLETS", "U_BULK_INLET", "U_MAX_INLETS", "U_MAX_INLET"),
        )
        reference_velocities = _expand_numeric_config_value(
            config_values,
            ("SA_REFERENCE_VELOCITIES", "SA_REFERENCE_VELOCITY"),
            inlet_count,
            default=reference_velocity_default,
        )
        reference_length_default = _first_config_value(
            config_values,
            (
                "INLET_WIDTHS",
                "INLET_WIDTH",
                "PORT_WIDTHS",
                "PORT_WIDTH",
                "HYDRAULIC_DIAMETERS",
                "HYDRAULIC_DIAMETER",
                "L",
            ),
        )
        reference_lengths = _expand_numeric_config_value(
            config_values,
            ("SA_REFERENCE_LENGTHS", "SA_REFERENCE_LENGTH"),
            inlet_count,
            default=reference_length_default,
        )

        ratio_targets = []
        for idx in range(inlet_count):
            ratio_targets.append(
                eddy_viscosity_ratio_from_turbulence_intensity(
                    intensities[idx],
                    nu_lam,
                    reference_velocity=(
                        reference_velocities[idx]
                        if reference_velocities is not None
                        else None
                    ),
                    reference_length=(
                        reference_lengths[idx]
                        if reference_lengths is not None
                        else None
                    ),
                    length_scale=(
                        length_scales[idx]
                        if length_scales is not None
                        else None
                    ),
                    length_scale_ratio=(
                        length_scale_ratios[idx]
                        if length_scale_ratios is not None
                        else None
                    ),
                    reynolds_number=(
                        reynolds_numbers[idx]
                        if reynolds_numbers is not None
                        else None
                    ),
                )
            )

        nu_tilde_targets = [nu_tilde_from_ratio(ratio, nu_lam) for ratio in ratio_targets]
        if length_scales is not None:
            length_text = "turbulence length scale ell = {}".format(
                _format_numeric_values(length_scales)
            )
        else:
            length_text = "turbulence length scale ratio ell/L_ref = {} (ell = inlet turbulence length scale)".format(
                _format_numeric_values(length_scale_ratios)
            )
        description = "turbulence intensity I = {}, {}, inferred eddy-viscosity ratio nu_t/nu_lam = {}".format(
            _format_numeric_values(intensities),
            length_text,
            _format_numeric_values(ratio_targets),
        )
        return nu_tilde_targets, description

    if "SA_NU_TILDE_INLETS" in config_values or "SA_NU_TILDE_INLET" in config_values:
        nu_tilde_targets = _expand_numeric_config_value(
            config_values,
            ("SA_NU_TILDE_INLETS", "SA_NU_TILDE_INLET"),
            inlet_count,
            required=True,
        )
        description = "direct nu_tilde = {}".format(_format_numeric_values(nu_tilde_targets))
        return nu_tilde_targets, description

    raise ValueError(
        "Define one of SA_TURBULENCE_INTENSITY, SA_MUT_RATIO, or SA_NU_TILDE_INLET."
    )


def create_design_mesh_from_config(config_values):
    mesh_builder = config_values.get("create_design_mesh")
    if callable(mesh_builder):
        return mesh_builder()
    raise ValueError(
        "Config module must define a callable create_design_mesh() to keep the Meshes + Utilities + Config workflow explicit."
    )


def _copy_xdmf_with_hdf_sidecars_to_temp(xdmf_path):
    """Copy an XDMF file and its HDF5 sidecars to a local temp directory."""
    tmp_dir = tempfile.mkdtemp(prefix="fenics_xdmf_", dir=tempfile.gettempdir())
    tmp_xdmf_path = os.path.join(tmp_dir, os.path.basename(xdmf_path))

    xdmf_dir = os.path.dirname(xdmf_path)
    shutil.copy2(xdmf_path, tmp_xdmf_path)

    with open(tmp_xdmf_path, "r", encoding="utf-8") as handle:
        xdmf_text = handle.read()

    h5_refs = sorted(set(re.findall(r">([^<>]+\\.h5):/", xdmf_text)))
    if not h5_refs:
        h5_refs = sorted(
            fname for fname in os.listdir(xdmf_dir)
            if fname.endswith(".h5")
        )

    for h5_name in h5_refs:
        shutil.copy2(os.path.join(xdmf_dir, h5_name), os.path.join(tmp_dir, h5_name))

    return tmp_xdmf_path


def load_mesh_from_xdmf(mesh_xdmf_path, comm=MPI.comm_world):
    mesh = Mesh()
    try:
        with XDMFFile(comm, mesh_xdmf_path) as xf:
            xf.read(mesh)
        return mesh
    except Exception as err:
        if MPI.rank(comm) == 0:
            print(
                "Warning: direct XDMF/HDF5 mesh read failed for {}. Retrying from the local temp directory.".format(
                    mesh_xdmf_path
                )
            )
            print("  Original read error: {}".format(err))

        tmp_xdmf_path = _copy_xdmf_with_hdf_sidecars_to_temp(mesh_xdmf_path)
        mesh_retry = Mesh()
        with XDMFFile(comm, tmp_xdmf_path) as xf:
            xf.read(mesh_retry)
        return mesh_retry


def load_cell_markers_from_xdmf(
    cell_xdmf_path,
    mesh,
    comm=MPI.comm_world,
    attribute_names=("name_to_read", "cell_tags"),
):
    """Read cell-wise integer markers from XDMF into a MeshFunction."""

    def _read_marker_function(xdmf_path):
        read_errors = []
        for attribute_name in as_list(attribute_names):
            mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
            try:
                with XDMFFile(comm, xdmf_path) as xf:
                    xf.read(mvc, attribute_name)
                return cpp.mesh.MeshFunctionSizet(mesh, mvc)
            except Exception as err:
                read_errors.append("{}: {}".format(attribute_name, err))
        raise RuntimeError(
            "Unable to read cell markers from {} using attributes [{}]. Errors: {}".format(
                xdmf_path,
                ", ".join(str(name) for name in as_list(attribute_names)),
                "; ".join(read_errors),
            )
        )

    try:
        return _read_marker_function(cell_xdmf_path)
    except Exception as err:
        if MPI.rank(comm) == 0:
            print(
                "Warning: direct XDMF/HDF5 cell-marker read failed for {}. "
                "Retrying from the local temp directory.".format(cell_xdmf_path)
            )
            print("  Original read error: {}".format(err))

        tmp_xdmf_path = _copy_xdmf_with_hdf_sidecars_to_temp(cell_xdmf_path)
        return _read_marker_function(tmp_xdmf_path)


def build_cell_tag_restriction_functions(
    cell_xdmf_path,
    design_tags,
    non_design_fluid_tags=(),
    non_design_solid_tags=(),
    attribute_names=("name_to_read", "cell_tags"),
):
    """Return config callbacks that derive DG0 bounds and regions from cell tags."""
    design_tag_set = {int(tag) for tag in as_list(design_tags)}
    non_design_fluid_tag_set = {int(tag) for tag in as_list(non_design_fluid_tags)}
    non_design_solid_tag_set = {int(tag) for tag in as_list(non_design_solid_tags)}

    overlaps = (
        design_tag_set & non_design_fluid_tag_set
        or design_tag_set & non_design_solid_tag_set
        or non_design_fluid_tag_set & non_design_solid_tag_set
    )
    if overlaps:
        raise ValueError(
            "Cell-tag restriction sets must be disjoint, got overlap {}.".format(
                sorted(overlaps)
            )
        )

    cache = {}

    def _build_restrictions(mesh, density_space):
        cache_key = (id(mesh), id(density_space))
        if cache_key in cache:
            return cache[cache_key]

        marker_function = load_cell_markers_from_xdmf(
            cell_xdmf_path,
            mesh,
            comm=mesh.mpi_comm(),
            attribute_names=attribute_names,
        )
        marker_values = marker_function.array()
        dofmap = density_space.dofmap()

        lower_bound = Function(density_space)
        upper_bound = Function(density_space)
        design_region = Function(density_space)

        lower_values = [0.0] * density_space.dim()
        upper_values = [1.0] * density_space.dim()
        region_values = [0.0] * density_space.dim()

        for cell in cells(mesh):
            cell_dofs = dofmap.cell_dofs(cell.index())
            if len(cell_dofs) != 1:
                raise ValueError(
                    "Cell-tag restrictions require one DG0 degree of freedom per cell."
                )

            dof = int(cell_dofs[0])
            tag = int(marker_values[cell.index()])

            if tag in design_tag_set:
                region_values[dof] = 1.0
            elif tag in non_design_fluid_tag_set:
                lower_values[dof] = 1.0
                upper_values[dof] = 1.0
            elif tag in non_design_solid_tag_set:
                lower_values[dof] = 0.0
                upper_values[dof] = 0.0
            else:
                raise ValueError(
                    "Encountered unexpected cell tag {} in {}. "
                    "Update the config tag mapping before running the optimization.".format(
                        tag, cell_xdmf_path
                    )
                )

        lower_bound.vector().set_local(lower_values)
        lower_bound.vector().apply("insert")
        upper_bound.vector().set_local(upper_values)
        upper_bound.vector().apply("insert")
        design_region.vector().set_local(region_values)
        design_region.vector().apply("insert")

        cache[cache_key] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "design_region": design_region,
        }
        return cache[cache_key]

    def _build_density_bounds(mesh, density_space):
        restriction_data = _build_restrictions(mesh, density_space)
        return restriction_data["lower_bound"], restriction_data["upper_bound"]

    def _build_volume_region(mesh, density_space):
        return _build_restrictions(mesh, density_space)["design_region"]

    def _build_objective_region(mesh, density_space):
        return _build_restrictions(mesh, density_space)["design_region"]

    return _build_density_bounds, _build_volume_region, _build_objective_region


def ensure_clean_dir(path, comm=MPI.comm_world):
    if MPI.rank(comm) == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    MPI.barrier(comm)


def reset_vtk_series(output_path, comm=MPI.comm_world):
    """Remove a stale PVD series before constructing a DOLFIN File writer."""
    if MPI.rank(comm) == 0:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        elif os.path.exists(output_path):
            os.remove(output_path)

        base_path, extension = os.path.splitext(output_path)
        if extension == ".pvd":
            for pattern in (base_path + "*.vtu", base_path + "*.pvtu"):
                matches = glob.glob(pattern)
                for match_path in matches:
                    if os.path.isdir(match_path):
                        shutil.rmtree(match_path)
                    elif os.path.exists(match_path):
                        os.remove(match_path)
    MPI.barrier(comm)
    return output_path


class ResilientVTKFile:
    """Try a shared-path VTK write first, then fall back to a local temp path."""

    def __init__(self, output_path, comm=MPI.comm_world, scratch_root=None):
        self._comm = comm
        self._primary_path = reset_vtk_series(output_path, comm)
        self._scratch_root = scratch_root or os.path.join(
            tempfile.gettempdir(), "fenics_vtk_outputs"
        )
        self._active_path = self._primary_path
        self._file = File(self._active_path)
        self._using_fallback = False
        self._disabled = False
        self._mirror_disabled = False

    def _build_fallback_path(self):
        relative_path = self._primary_path.lstrip(os.sep)
        return os.path.join(self._scratch_root, relative_path)

    def _switch_to_fallback(self, write_error):
        fallback_path = self._build_fallback_path()
        self._file = File(reset_vtk_series(fallback_path, self._comm))
        self._active_path = fallback_path
        self._using_fallback = True
        if MPI.rank(self._comm) == 0:
            print(
                "Warning: DOLFIN could not write VTK output to {}. Switching VTK output to {}.".format(
                    self._primary_path,
                    fallback_path,
                )
            )
            print("  Original VTK write error: {}".format(write_error))

    def _disable_writes(self, write_error):
        self._disabled = True
        if MPI.rank(self._comm) == 0:
            print(
                "Warning: DOLFIN could not write VTK output to {}. Disabling further writes for this series.".format(
                    self._active_path
                )
            )
            print("  Final VTK write error: {}".format(write_error))

    def _mirror_fallback_to_primary(self):
        if (not self._using_fallback) or self._mirror_disabled:
            return

        MPI.barrier(self._comm)
        if MPI.rank(self._comm) != 0:
            MPI.barrier(self._comm)
            return

        try:
            primary_dir = os.path.dirname(self._primary_path)
            if primary_dir:
                os.makedirs(primary_dir, exist_ok=True)

            active_base, extension = os.path.splitext(self._active_path)
            files_to_copy = []
            if os.path.exists(self._active_path):
                files_to_copy.append(self._active_path)
            if extension == ".pvd":
                for pattern in (active_base + "*.vtu", active_base + "*.pvtu"):
                    files_to_copy.extend(glob.glob(pattern))

            for source_path in files_to_copy:
                destination_path = os.path.join(primary_dir, os.path.basename(source_path))
                shutil.copy2(source_path, destination_path)
        except OSError as err:
            self._mirror_disabled = True
            print(
                "Warning: fallback VTK output was written to {}, but mirroring it back to {} failed.".format(
                    self._active_path,
                    self._primary_path,
                )
            )
            print("  Mirror error: {}".format(err))
        finally:
            MPI.barrier(self._comm)

    def __lshift__(self, other):
        if self._disabled:
            return self
        try:
            self._file << other
            self._mirror_fallback_to_primary()
        except RuntimeError as err:
            if self._using_fallback:
                self._disable_writes(err)
                return self
            self._switch_to_fallback(err)
            try:
                self._file << other
                self._mirror_fallback_to_primary()
            except RuntimeError as fallback_err:
                self._disable_writes(fallback_err)
        return self

def compute_filter_base_length_from_config(config_values):
    custom_base = config_values.get("FILTER_BASE_LENGTH")
    if custom_base is not None:
        return float(custom_base)

    if "L" in config_values and "N" in config_values:
        return float(config_values["L"]) / float(config_values["N"])

    x_min = float(config_values.get("DOMAIN_X_MIN", 0.0))
    y_min = float(config_values.get("DOMAIN_Y_MIN", 0.0))
    x_max = float(config_values.get("DOMAIN_X_MAX", x_min + 1.0))
    y_max = float(config_values.get("DOMAIN_Y_MAX", y_min + 1.0))
    nx = int(config_values.get("NX", config_values.get("N", 1)))
    ny = int(config_values.get("NY", config_values.get("N", 1)))
    return min((x_max - x_min) / float(max(nx, 1)), (y_max - y_min) / float(max(ny, 1)))


def build_pressure_pin_expression_from_config(config_values):
    if "PRESSURE_PIN_POINT" in config_values:
        pin_x, pin_y = config_values["PRESSURE_PIN_POINT"]
    else:
        pin_x = float(config_values.get("DOMAIN_X_MIN", 0.0))
        pin_y = float(config_values.get("DOMAIN_Y_MIN", 0.0))
    return "near(x[0], {:.16g}) && near(x[1], {:.16g})".format(float(pin_x), float(pin_y))


def area_weighted_boundary_average(field, boundary_measure, markers, area_floor=1.0e-14):
    marker_list = as_list(markers)
    one = Constant(1.0)
    weighted_integral = 0.0
    boundary_area = 0.0
    for marker in marker_list:
        weighted_integral += assemble(field * boundary_measure(marker))
        boundary_area += assemble(one * boundary_measure(marker))
    boundary_area = float(boundary_area)
    if boundary_area <= area_floor:
        return float("nan")
    return float(weighted_integral) / boundary_area


def pressure_drop_between_boundaries(pressure, boundary_measure, inlet_markers, outlet_markers):
    inlet_pressure = area_weighted_boundary_average(pressure, boundary_measure, inlet_markers)
    outlet_pressure = area_weighted_boundary_average(pressure, boundary_measure, outlet_markers)
    return inlet_pressure - outlet_pressure


def area_weighted_internal_facet_average(field, facet_measure, markers, area_floor=1.0e-14):
    marker_list = as_list(markers)
    one = Constant(1.0)
    weighted_integral = 0.0
    boundary_area = 0.0
    for marker in marker_list:
        weighted_integral += assemble(avg(field) * facet_measure(marker))
        boundary_area += assemble(avg(one) * facet_measure(marker))
    boundary_area = float(boundary_area)
    if boundary_area <= area_floor:
        return float("nan")
    return float(weighted_integral) / boundary_area


def pressure_drop_between_internal_facets(pressure, facet_measure, inlet_markers, outlet_markers):
    inlet_pressure = area_weighted_internal_facet_average(pressure, facet_measure, inlet_markers)
    outlet_pressure = area_weighted_internal_facet_average(pressure, facet_measure, outlet_markers)
    return inlet_pressure - outlet_pressure


class _PlaneSegmentSubDomain(SubDomain):
    def __init__(self, axis, location, range_axis, lower, upper, tol):
        super().__init__()
        self._axis = int(axis)
        self._location = float(location)
        self._range_axis = int(range_axis)
        self._lower = float(lower)
        self._upper = float(upper)
        self._tol = float(tol)

    def inside(self, x, on_boundary):
        return near(x[self._axis], self._location, self._tol) and (
            self._lower - self._tol <= x[self._range_axis] <= self._upper + self._tol
        )


def _plane_spec(axis, location, lower, upper):
    axis = int(axis)
    if axis not in (0, 1):
        raise ValueError("Only 2D pressure-drop planes are supported.")
    return {
        "axis": axis,
        "location": float(location),
        "range_axis": 1 - axis,
        "lower": float(lower),
        "upper": float(upper),
    }


def _normalize_pressure_plane_specs(raw_specs):
    normalized = []
    for raw_spec in as_list(raw_specs):
        if not isinstance(raw_spec, dict):
            raise TypeError("Pressure-drop plane specifications must be dictionaries.")
        axis = int(raw_spec["axis"])
        range_axis = int(raw_spec.get("range_axis", 1 - axis))
        lower, upper = raw_spec["range"]
        normalized.append({
            "axis": axis,
            "location": float(raw_spec["location"]),
            "range_axis": range_axis,
            "lower": float(lower),
            "upper": float(upper),
        })
    return normalized


def _infer_design_pressure_drop_plane_specs(config_values):
    if "DESIGN_PRESSURE_DROP_PLANES" in config_values:
        planes = config_values["DESIGN_PRESSURE_DROP_PLANES"]
        return (
            _normalize_pressure_plane_specs(planes["inlet"]),
            _normalize_pressure_plane_specs(planes["outlet"]),
        )

    design_x_min = float(config_values.get("DESIGN_X_MIN", config_values.get("DOMAIN_X_MIN", 0.0)))
    design_x_max = float(config_values.get("DESIGN_X_MAX", config_values.get("DOMAIN_X_MAX", 1.0)))
    design_y_min = float(config_values.get("DESIGN_Y_MIN", config_values.get("DOMAIN_Y_MIN", 0.0)))
    domain_y_min = float(config_values.get("DOMAIN_Y_MIN", design_y_min))
    domain_y_max = float(config_values.get("DOMAIN_Y_MAX", config_values.get("DESIGN_Y_MAX", 1.0)))

    if "INLET_SEGMENTS" in config_values and "OUTLET_SEGMENTS" in config_values:
        inlet_specs = [
            _plane_spec(0, design_x_min, segment[0], segment[1])
            for segment in as_list(config_values["INLET_SEGMENTS"])
        ]
        outlet_specs = [
            _plane_spec(0, design_x_max, segment[0], segment[1])
            for segment in as_list(config_values["OUTLET_SEGMENTS"])
        ]
        return inlet_specs, outlet_specs

    if all(name in config_values for name in ("TOP_PORT_Y_MIN", "TOP_PORT_Y_MAX", "BOTTOM_PORT_Y_MIN", "BOTTOM_PORT_Y_MAX")):
        return (
            [_plane_spec(0, design_x_min, config_values["TOP_PORT_Y_MIN"], config_values["TOP_PORT_Y_MAX"])],
            [_plane_spec(0, design_x_min, config_values["BOTTOM_PORT_Y_MIN"], config_values["BOTTOM_PORT_Y_MAX"])],
        )

    if all(name in config_values for name in ("INLET_Y_MIN", "INLET_Y_MAX", "OUTLET_X_MIN", "OUTLET_X_MAX")):
        return (
            [_plane_spec(0, design_x_min, config_values["INLET_Y_MIN"], config_values["INLET_Y_MAX"])],
            [_plane_spec(1, design_y_min, config_values["OUTLET_X_MIN"], config_values["OUTLET_X_MAX"])],
        )

    if "OUTLET_Y_MIN" in config_values and "OUTLET_Y_MAX" in config_values:
        return (
            [_plane_spec(0, design_x_min, domain_y_min, domain_y_max)],
            [_plane_spec(0, design_x_max, config_values["OUTLET_Y_MIN"], config_values["OUTLET_Y_MAX"])],
        )

    raise ValueError(
        "Unable to infer design-domain pressure-drop planes. Define "
        "DESIGN_PRESSURE_DROP_PLANES in the config."
    )


def build_design_pressure_drop_markers(
    mesh,
    config_values,
    inlet_marker=101,
    outlet_marker=102,
):
    inlet_specs, outlet_specs = _infer_design_pressure_drop_plane_specs(config_values)
    tol = float(config_values.get("PRESSURE_DROP_PLANE_TOL", config_values.get("TOL", 1.0e-12)))
    facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_markers.set_all(0)

    for spec in inlet_specs:
        _PlaneSegmentSubDomain(
            spec["axis"], spec["location"], spec["range_axis"], spec["lower"], spec["upper"], tol
        ).mark(facet_markers, int(inlet_marker))

    for spec in outlet_specs:
        _PlaneSegmentSubDomain(
            spec["axis"], spec["location"], spec["range_axis"], spec["lower"], spec["upper"], tol
        ).mark(facet_markers, int(outlet_marker))

    return facet_markers, {"inlet": int(inlet_marker), "outlet": int(outlet_marker)}


def _optimization_log_columns(
    include_ipcs_residuals=False,
    pressure_drop_columns=None,
    objective_column="Objective",
    extra_columns=None,
):
    if pressure_drop_columns is None:
        pressure_drop_columns = ("DeltaP_Pa",)
    if extra_columns is None:
        extra_columns = ()
    objective_column = str(objective_column)

    columns = [
        ("Stage", 7),
        ("Q", 7),
        ("Beta", 7),
        ("InnerIter", 10),
        ("GlobalIter", 11),
        (objective_column, max(17, len(objective_column))),
    ]
    columns.extend((str(name), max(17, len(str(name)))) for name in pressure_drop_columns)
    columns.extend([
        ("ObjConv", 17),
        ("VolFrac", 17),
        ("VolResid", 17),
    ])
    columns.extend((str(name), max(17, len(str(name)))) for name in extra_columns)
    columns.append(("Timestamp", 24))
    return columns


def _format_optimization_log_row(
    values,
    include_ipcs_residuals=False,
    pressure_drop_columns=None,
    objective_column="Objective",
    extra_columns=None,
):
    columns = _optimization_log_columns(
        include_ipcs_residuals=include_ipcs_residuals,
        pressure_drop_columns=pressure_drop_columns,
        objective_column=objective_column,
        extra_columns=extra_columns,
    )
    if len(values) != len(columns):
        raise ValueError(
            "Expected {} optimization-log fields, got {}.".format(len(columns), len(values))
        )
    return "   ".join(str(value).ljust(width) for value, (_, width) in zip(values, columns))


def initialize_optimization_log(
    log_path,
    include_ipcs_residuals=False,
    pressure_drop_columns=None,
    objective_column="Objective",
    extra_columns=None,
):
    header_fields = _optimization_log_columns(
        include_ipcs_residuals=include_ipcs_residuals,
        pressure_drop_columns=pressure_drop_columns,
        objective_column=objective_column,
        extra_columns=extra_columns,
    )
    if MPI.rank(MPI.comm_world) == 0:
        with open(log_path, "w") as txtout:
            txtout.write(_format_optimization_log_row(
                [label for label, _ in header_fields],
                include_ipcs_residuals=include_ipcs_residuals,
                pressure_drop_columns=pressure_drop_columns,
                objective_column=objective_column,
                extra_columns=extra_columns,
            ) + "\r\n")
    MPI.barrier(MPI.comm_world)


def append_optimization_log_entry(
    log_path,
    stage_idx,
    q_value,
    beta_value,
    inner_iter,
    global_iter,
    objective,
    pressure_drop,
    objective_convergence,
    volume_fraction,
    volume_residual,
    pressure_drop_values=None,
    du_ipcs=None,
    dp_ipcs=None,
    pressure_drop_columns=None,
    objective_column="Objective",
    extra_values=None,
    extra_columns=None,
):
    if MPI.rank(MPI.comm_world) == 0:
        with open(log_path, "a") as txtout:
            if pressure_drop_values is None:
                pressure_drop_values = [pressure_drop]
            pressure_drop_values = [float(value) for value in pressure_drop_values]
            if pressure_drop_columns is None:
                pressure_drop_columns = ["dP"] * len(pressure_drop_values)
            if extra_values is None:
                extra_values = []
            if extra_columns is None:
                extra_columns = []
            extra_values = [float(value) for value in extra_values]
            if len(extra_values) != len(extra_columns):
                raise ValueError(
                    "Expected {} extra optimization-log values, got {}.".format(
                        len(extra_columns), len(extra_values)
                    )
                )
            row_values = [
                "{:d}".format(int(stage_idx)),
                "{:.3f}".format(float(q_value)),
                "{:.2f}".format(float(beta_value)),
                "{:d}".format(int(inner_iter)),
                "{:d}".format(int(global_iter)),
                "{:.10e}".format(float(objective)),
            ]
            row_values.extend("{:.10e}".format(value) for value in pressure_drop_values)
            row_values.extend([
                "{:.10e}".format(float(objective_convergence)),
                "{:.10e}".format(float(volume_fraction)),
                "{:.10e}".format(float(volume_residual)),
            ])
            row_values.extend("{:.10e}".format(value) for value in extra_values)
            row_values.append(strftime("%a, %d %b %Y %H:%M:%S", localtime()))
            txtout.write(
                _format_optimization_log_row(
                    row_values,
                    include_ipcs_residuals=(du_ipcs is not None or dp_ipcs is not None),
                    pressure_drop_columns=pressure_drop_columns,
                    objective_column=objective_column,
                    extra_columns=extra_columns,
                ) + "\r\n"
            )
