"""Lightweight helpers for reading grid and observation paths from visualise_config.toml.

Kept dependency-free (stdlib only) so modules that must not pull in the heavy
plotting stack (cartopy/gsw/matplotlib via map_utils) can still resolve the
per-run mask/mesh and observation paths. See map_utils.py for the plotting code
that uses it.

The active config is selected per-run via $VISUALISE_CONFIG (NEMO3.6 vs NEMO5);
there is no NEMO-version default, so a misconfigured run fails loudly instead of
silently using the wrong grid or wrong-grid obs.
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


def _load_files():
    """Return (config_path, [files] table) from the active visualise_config.toml.

    Raises FileNotFoundError if no config is found — there is no NEMO-version
    default, so a misconfigured run fails loudly rather than guessing the grid/obs.
    """
    cfg = _find_visualise_config()
    if cfg is None:
        raise FileNotFoundError(
            "visualise_config.toml not found (checked $VISUALISE_CONFIG, cwd, and "
            f"{Path(__file__).parent}). Grid mask/mesh and observation paths come "
            "from its [files] section; there is no NEMO-version default."
        )
    with open(cfg, "rb") as f:
        return cfg, tomllib.load(f).get("files", {})


# Runs created before visualise_config: was added to setUpData are all NEMO3.6
# (the field arrived with NEMO5 dual-grid support), so old-style setUpData falls
# back to the original 3.6 grid/obs config to stay visualisable.
LEGACY_CONFIG_NAME = "visualise_config_nemo3.6.toml"


def resolve_run_config(run_dir):
    """Resolve the visualise_config path for a model run from its setUpData.

    Reads ``visualise_config:`` from ``<run_dir>/setUpData_*.dat`` and resolves it
    the way setUpRun.sh does: an absolute path is used as-is, else it is taken
    relative to this visualise/ directory. An old-style setUpData that predates the
    field falls back to the legacy NEMO3.6 config. Returns None only when no
    setUpData is present at all.
    """
    run_dir = Path(run_dir)
    dats = sorted(run_dir.glob("setUpData_*.dat"))
    if not dats:
        return None
    value = ""
    with open(dats[0]) as f:
        for line in f:
            if line.strip().startswith("visualise_config:"):
                value = line.split(":", 1)[1].strip()
                break
    if not value:
        value = LEGACY_CONFIG_NAME  # old-style setUpData: pre-field, so NEMO3.6
    p = Path(value)
    return p if p.is_absolute() else Path(__file__).parent / value


def load_config_for_runs(run_dirs):
    """Load the one shared visualise config for a set of model runs (no env vars).

    Each run's config is taken from its setUpData (resolve_run_config); a
    multi-model comparison shares one grid, so every run must name the same
    config. Returns the parsed config dict. Raises FileNotFoundError if no run
    carries the field and ValueError on a mismatch -- there is no NEMO-version
    default, so a misconfigured comparison fails loudly rather than guessing.
    """
    resolved = {}
    for d in run_dirs:
        p = resolve_run_config(d)
        if p is not None:
            resolved[Path(d).name] = p

    if not resolved:
        raise FileNotFoundError(
            "no setUpData_*.dat found in any run directory; cannot resolve the "
            "grid/obs config (old-style setUpData without the visualise_config: "
            "line falls back to NEMO3.6, but the file itself must be present)."
        )
    if len({str(p) for p in resolved.values()}) > 1:
        details = ", ".join(f"{n}={p.name}" for n, p in resolved.items())
        raise ValueError(
            f"runs name different visualise_config files ({details}); a comparison "
            "shares one grid. Align visualise_config: in their setUpData."
        )

    cfg = next(iter(resolved.values()))
    if not cfg.is_file():
        raise FileNotFoundError(f"visualise_config from setUpData not found: {cfg}")
    with open(cfg, "rb") as f:
        return tomllib.load(f)


def get_mask_paths():
    """Return (basin_mask, mesh_mask) from visualise_config.toml [files].

    The grid is set per-run via the selected config (e.g. NEMO3.6 vs NEMO5);
    there is no NEMO-version default.
    """
    cfg, files = _load_files()
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


def get_obs_dir(override=None):
    """Base observations directory: ``override`` if given, else [files].obs_dir.

    ``override`` is a script's ``--obs-dir`` (None when not passed), letting the
    CLI redirect the directory while the filenames stay config-driven.
    """
    if override:
        return str(override)
    cfg, files = _load_files()
    try:
        return files["obs_dir"]
    except KeyError:
        raise KeyError(
            f"{cfg} is missing [files].obs_dir; set it per-run or pass --obs-dir."
        ) from None


def get_obs_filenames():
    """Mapping {logical_name: path-relative-to-obs_dir} from [files.obs]."""
    cfg, files = _load_files()
    obs = files.get("obs", {})
    if not obs:
        raise KeyError(
            f"{cfg} is missing the [files.obs] table of observation files."
        )
    return obs


def get_obs_path(key, obs_dir=None):
    """Absolute path to observation file ``key`` (from [files.obs]).

    The directory is ``obs_dir`` if given (a script's --obs-dir) else the config's
    [files].obs_dir; the filename is [files.obs][key]. Raises KeyError on an
    unknown key so a typo fails loudly rather than silently skipping an obs panel.
    """
    names = get_obs_filenames()
    try:
        rel = names[key]
    except KeyError:
        raise KeyError(
            f"unknown observation key '{key}'; [files.obs] defines: "
            f"{', '.join(sorted(names))}"
        ) from None
    return str(Path(get_obs_dir(obs_dir)) / rel)
