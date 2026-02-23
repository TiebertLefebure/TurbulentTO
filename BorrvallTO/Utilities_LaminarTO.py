import argparse
import importlib
import os
import shutil

from dolfin import MPI


def normalize_module_name(module_name):
    normalized = module_name.strip()
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    normalized = normalized.replace(os.sep, ".")
    return normalized


def load_config_module_from_cli():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        required=True,
        help="Python module name for solver configuration (required).",
    )
    args, _unknown = parser.parse_known_args()
    module_name = normalize_module_name(args.config)
    if not module_name:
        raise ValueError("Empty config module name is not allowed.")
    return module_name, importlib.import_module(module_name)


def float_scalar(value):
    if hasattr(value, "values"):
        values = value.values()
        if len(values) == 1:
            return float(values[0])
    return float(value)


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def match_count(values, target_size):
    values = as_list(values)
    if len(values) == target_size:
        return values
    if len(values) == 1:
        return values * target_size
    if len(values) < target_size:
        return values + [values[-1]] * (target_size - len(values))
    return values[:target_size]


def expand_to_match(values, target_size, label):
    values = as_list(values)
    if len(values) == target_size:
        return values
    if len(values) == 1 and target_size > 1:
        return values * target_size
    raise ValueError(
        "{} count ({}) must match marker count ({})".format(label, len(values), target_size)
    )


def ensure_clean_dir(path, comm=MPI.comm_world):
    # Avoid MPI races where multiple ranks delete/create the same folder.
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


def compute_legacy_port_extents_from_config(config_values):
    port_extent_builder = config_values.get("compute_port_extents")
    if not callable(port_extent_builder):
        return None

    required = [
        "L",
        "INLET_TOP_OFFSET",
        "INLET_WIDTH",
        "OUTLET_RIGHT_OFFSET",
        "OUTLET_WIDTH",
    ]
    if any(name not in config_values for name in required):
        return None

    return port_extent_builder(
        config_values["L"],
        config_values["INLET_TOP_OFFSET"],
        config_values["INLET_WIDTH"],
        config_values["OUTLET_RIGHT_OFFSET"],
        config_values["OUTLET_WIDTH"],
    )


def build_pressure_pin_expression_from_config(config_values):
    if "PRESSURE_PIN_POINT" in config_values:
        pin_x, pin_y = config_values["PRESSURE_PIN_POINT"]
    else:
        pin_x = float(config_values.get("DOMAIN_X_MIN", 0.0))
        pin_y = float(config_values.get("DOMAIN_Y_MIN", 0.0))
    return "near(x[0], {:.16g}) && near(x[1], {:.16g})".format(float(pin_x), float(pin_y))
