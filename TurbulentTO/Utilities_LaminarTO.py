import argparse
import importlib
import os
import re
import shutil
import subprocess
import tempfile
from time import localtime, strftime

from dolfin import MPI, Mesh, XDMFFile


def load_config_module_from_cli():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        required=True,
        help="Python module name for solver configuration (required).",
    )
    args, _unknown = parser.parse_known_args()
    config_str = args.config.strip()
    if config_str.endswith(".py"):
        config_str = config_str[:-3]
    module_name = config_str.replace(os.sep, ".")
    if not module_name:
        raise ValueError("Empty config module name is not allowed.")
    return module_name, importlib.import_module(module_name)


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


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

        def _copy_to_tmp(src_path, dst_path):
            try:
                shutil.copy2(src_path, dst_path)
            except Exception:
                # Some host-mounted filesystems exposed inside containers can fail on
                # direct Python reads while shell-level copy still succeeds.
                subprocess.check_call(["cp", src_path, dst_path])

        xdmf_dir = os.path.dirname(mesh_xdmf_path)
        _copy_to_tmp(mesh_xdmf_path, tmp_xdmf_path)

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
            _copy_to_tmp(os.path.join(xdmf_dir, h5_name), os.path.join(tmp_dir, h5_name))

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


def initialize_optimization_log(log_path):
    header_fields = (
        ("Stage", 7),
        ("Q", 7),
        ("Beta", 7),
        ("InnerIter", 10),
        ("GlobalIter", 11),
        ("Objective", 14),
        ("ObjConv", 12),
        ("VolFrac", 12),
        ("VolResid", 12),
        ("Timestamp", 24),
    )
    with open(log_path, "w") as txtout:
        txtout.write(
            " ".join(label.ljust(width) for label, width in header_fields) + "\r\n"
        )


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
):
    with open(log_path, "a") as txtout:
        txtout.write(
            "{:<7d} {:<7.3f} {:<7.2f} {:<10d} {:<11d} {:.10e}   {:.10e}   {:.10e}   {:.10e}   {}\r\n".format(
                int(stage_idx),
                float(q_value),
                float(beta_value),
                int(inner_iter),
                int(global_iter),
                float(objective),
                float(objective_convergence),
                float(volume_fraction),
                float(volume_residual),
                strftime("%a, %d %b %Y %H:%M:%S", localtime()),
            )
        )
