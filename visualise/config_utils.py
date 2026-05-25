"""Lightweight helpers for reading grid paths from visualise_config.toml.

Kept dependency-free (stdlib only) so modules that must not pull in the heavy
plotting stack (cartopy/gsw/matplotlib via map_utils) can still resolve the
per-run mask/mesh paths. See map_utils.py for the plotting code that uses it.
"""

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _find_visualise_config():
    """Locate visualise_config.toml: $VISUALISE_CONFIG, then cwd, then this dir."""
    env = os.environ.get("VISUALISE_CONFIG", "")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path("visualise_config.toml"))
    candidates.append(Path(__file__).parent / "visualise_config.toml")
    for p in candidates:
        if p.is_file():
            return p
    return None


def get_mask_paths():
    """Return (basin_mask, mesh_mask) from visualise_config.toml [files].

    The grid is set per-run via the selected config (e.g. NEMO3.6 vs NEMO5);
    there is no NEMO-version default. Raises if the config or the [files] paths
    are missing, so a misconfigured run fails loudly rather than using the wrong
    grid.
    """
    cfg = _find_visualise_config()
    if cfg is None:
        raise FileNotFoundError(
            "visualise_config.toml not found (checked $VISUALISE_CONFIG, cwd, and "
            f"{Path(__file__).parent}). The grid mask/mesh paths come from its "
            "[files] section; there is no NEMO-version default."
        )
    with open(cfg, "rb") as f:
        files = tomllib.load(f).get("files", {})
    try:
        return files["basin_mask"], files["mesh_mask"]
    except KeyError as exc:
        raise KeyError(
            f"{cfg} is missing [files].{exc.args[0]}; grid mask/mesh paths must be "
            "set per-run (no NEMO-version default)."
        ) from None


def get_basin_mask_path():
    """basin_mask path from visualise_config.toml [files]."""
    return get_mask_paths()[0]


def get_mesh_mask_path():
    """mesh_mask path from visualise_config.toml [files]."""
    return get_mask_paths()[1]
