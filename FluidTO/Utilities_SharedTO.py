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

from dolfin import File, MPI, Mesh, XDMFFile


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

    If reynolds_number is supplied, length_scale_ratio is interpreted as ell/L.
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
            length_text = "ell = {}".format(_format_numeric_values(length_scales))
        else:
            length_text = "ell/L = {}".format(_format_numeric_values(length_scale_ratios))
        description = "I = {}, {}, inferred nu_t/nu_lam = {}".format(
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


def load_mesh_from_xdmf(mesh_xdmf_path, comm=MPI.comm_world):
    mesh = Mesh()
    try:
        with XDMFFile(comm, mesh_xdmf_path) as xf:
            xf.read(mesh)
        return mesh
    except Exception as err:
        if MPI.rank(comm) == 0:
            print(
                "Warning: direct XDMF/HDF5 mesh read failed for {}. Retrying from /tmp.".format(
                    mesh_xdmf_path
                )
            )
            print("  Original read error: {}".format(err))

        tmp_dir = tempfile.mkdtemp(prefix="fenics_xdmf_", dir="/tmp")
        tmp_xdmf_path = os.path.join(tmp_dir, os.path.basename(mesh_xdmf_path))

        xdmf_dir = os.path.dirname(mesh_xdmf_path)
        shutil.copy2(mesh_xdmf_path, tmp_xdmf_path)

        # Read the local tmp copy, not the mounted source file, to discover HDF
        # sidecars without re-triggering shared-folder IO issues.
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

        mesh_retry = Mesh()
        with XDMFFile(comm, tmp_xdmf_path) as xf:
            xf.read(mesh_retry)
        return mesh_retry


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
    """Try a shared-path VTK write first, then fall back to a local /tmp path."""

    def __init__(self, output_path, comm=MPI.comm_world, scratch_root=None):
        self._comm = comm
        self._primary_path = reset_vtk_series(output_path, comm)
        self._scratch_root = scratch_root or os.path.join("/tmp", "fenics_vtk_outputs")
        self._active_path = self._primary_path
        self._file = File(self._active_path)
        self._using_fallback = False
        self._disabled = False

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

    def __lshift__(self, other):
        if self._disabled:
            return self
        try:
            self._file << other
        except RuntimeError as err:
            if self._using_fallback:
                self._disable_writes(err)
                return self
            self._switch_to_fallback(err)
            try:
                self._file << other
            except RuntimeError as fallback_err:
                self._disable_writes(fallback_err)
        return self

    @property
    def active_path(self):
        return self._active_path


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


def _optimization_log_columns(include_ipcs_residuals=False):
    columns = [
        ("Stage", 7),
        ("Q", 7),
        ("Beta", 7),
        ("InnerIter", 10),
        ("GlobalIter", 11),
        ("Objective", 17),
        ("ObjConv", 17),
        ("VolFrac", 17),
        ("VolResid", 17),
    ]
    if include_ipcs_residuals:
        columns.extend([
            ("du_ipcs", 17),
            ("dp_ipcs", 17),
        ])
    columns.append(("Timestamp", 24))
    return columns


def _format_optimization_log_row(values, include_ipcs_residuals=False):
    columns = _optimization_log_columns(include_ipcs_residuals=include_ipcs_residuals)
    if len(values) != len(columns):
        raise ValueError(
            "Expected {} optimization-log fields, got {}.".format(len(columns), len(values))
        )
    return "   ".join(str(value).ljust(width) for value, (_, width) in zip(values, columns))


def initialize_optimization_log(log_path, include_ipcs_residuals=False):
    header_fields = _optimization_log_columns(include_ipcs_residuals=include_ipcs_residuals)
    if MPI.rank(MPI.comm_world) == 0:
        with open(log_path, "w") as txtout:
            txtout.write(_format_optimization_log_row(
                [label for label, _ in header_fields],
                include_ipcs_residuals=include_ipcs_residuals,
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
    objective_convergence,
    volume_fraction,
    volume_residual,
    du_ipcs=None,
    dp_ipcs=None,
):
    if MPI.rank(MPI.comm_world) == 0:
        with open(log_path, "a") as txtout:
            row_values = [
                "{:d}".format(int(stage_idx)),
                "{:.3f}".format(float(q_value)),
                "{:.2f}".format(float(beta_value)),
                "{:d}".format(int(inner_iter)),
                "{:d}".format(int(global_iter)),
                "{:.10e}".format(float(objective)),
                "{:.10e}".format(float(objective_convergence)),
                "{:.10e}".format(float(volume_fraction)),
                "{:.10e}".format(float(volume_residual)),
            ]
            if du_ipcs is not None or dp_ipcs is not None:
                if du_ipcs is None or dp_ipcs is None:
                    raise ValueError("Both du_ipcs and dp_ipcs must be provided together.")
                row_values.extend([
                    "{:.10e}".format(float(du_ipcs)),
                    "{:.10e}".format(float(dp_ipcs)),
                ])
            row_values.append(strftime("%a, %d %b %Y %H:%M:%S", localtime()))
            txtout.write(
                _format_optimization_log_row(
                    row_values,
                    include_ipcs_residuals=(du_ipcs is not None or dp_ipcs is not None),
                ) + "\r\n"
            )
