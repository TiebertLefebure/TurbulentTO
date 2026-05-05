import os
import tempfile


def _is_shared_container_path(path):
    """Return True for Docker shared mounts that are unsafe for FEniCS JIT files."""
    if path is None:
        return False
    normalized = os.path.abspath(path)
    shared_roots = ("/home/fenics/shared", "/root/shared")
    return any(normalized == root or normalized.startswith(root + os.sep) for root in shared_roots)


def _tmp_runtime_root():
    repo_name = os.path.basename(os.getcwd()) or "turbulentto"
    try:
        user_id = os.getuid()
    except AttributeError:
        user_id = "user"
    return os.path.join(tempfile.gettempdir(), "{}_runtime_{}".format(repo_name, user_id))


def configure_writable_runtime_environment(base_dir=None):
    """Force JIT/temp files into a writable project-local runtime directory."""
    runtime_override = os.environ.get("TURBULENTTO_RUNTIME_ROOT")
    runtime_base = base_dir
    if runtime_override is None and _is_shared_container_path(runtime_base):
        runtime_base = _tmp_runtime_root()
    runtime_root = os.path.abspath(
        runtime_override
        or runtime_base
        or os.path.join(os.getcwd(), ".runtime")
    )
    tmp_dir = os.path.join(runtime_root, "tmp")
    cache_root = os.path.join(runtime_root, "cache")
    dijitso_cache_dir = os.path.join(cache_root, "dijitso")
    instant_cache_dir = os.path.join(cache_root, "instant")

    for path in (runtime_root, tmp_dir, cache_root, dijitso_cache_dir, instant_cache_dir):
        os.makedirs(path, exist_ok=True)

    for env_name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[env_name] = tmp_dir

    os.environ["XDG_CACHE_HOME"] = cache_root
    os.environ["DIJITSO_CACHE_DIR"] = dijitso_cache_dir
    os.environ["INSTANT_CACHE_DIR"] = instant_cache_dir
    tempfile.tempdir = tmp_dir

    return runtime_root
