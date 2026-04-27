import os
import tempfile


def configure_writable_runtime_environment(base_dir=None):
    """Force JIT/temp files into a writable project-local runtime directory."""
    runtime_root = os.path.abspath(
        base_dir
        or os.environ.get("TURBULENTTO_RUNTIME_ROOT")
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
