"""Helpers for reading grid and observation paths from a run's visualise_config.toml.

Stdlib-only so modules that must not pull in the heavy plotting stack
(cartopy/gsw/matplotlib via map_utils) can still resolve the per-run mask/mesh and
observation paths. See map_utils.py for the plotting code that uses it.

The active config is selected per-run from the run's ``setUpData_*.dat``
(``visualise_config:``), the same way setUpRun.sh resolves it (NEMO3.6 vs NEMO5).
There is no NEMO-version default and no environment-variable or cwd discovery:
callers resolve the config from a run directory (``load_config`` /
``load_config_for_runs``) and thread the parsed dict explicitly into the accessor
helpers below. A misconfigured run therefore fails loudly instead of silently
using the wrong grid or wrong-grid obs.
"""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


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


def load_config(run_dir=None, *, config_path=None):
    """Parse the visualise_config for a single run.

    Provide exactly one of ``run_dir`` (resolved from its setUpData via
    resolve_run_config) or ``config_path`` (an explicit file). There is no
    ambient discovery, so a missing config fails loudly rather than guessing.
    """
    if config_path is not None:
        p = Path(config_path)
    elif run_dir is not None:
        p = resolve_run_config(run_dir)
        if p is None:
            raise FileNotFoundError(
                f"no setUpData_*.dat in {run_dir}; cannot resolve visualise_config "
                "(grid mask/mesh and observation paths come from it)."
            )
    else:
        raise TypeError("load_config requires run_dir or config_path")
    if not p.is_file():
        raise FileNotFoundError(f"visualise_config not found: {p}")
    with open(p, "rb") as f:
        return tomllib.load(f)


def load_config_for_runs(run_dirs):
    """Load the one shared visualise config for a set of model runs.

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


# ---------------------------------------------------------------------------
# Accessors -- all take the parsed config dict (from load_config*) explicitly.
# ---------------------------------------------------------------------------

def _files(config):
    return config.get("files", {})


def get_basin_mask_path(config):
    """basin_mask path from a loaded config's [files]."""
    try:
        return _files(config)["basin_mask"]
    except KeyError:
        raise KeyError(
            "visualise_config [files].basin_mask is missing; the grid mask path "
            "must be set per-run (no NEMO-version default)."
        ) from None


def get_mesh_mask_path(config):
    """mesh_mask path from a loaded config's [files]."""
    try:
        return _files(config)["mesh_mask"]
    except KeyError:
        raise KeyError(
            "visualise_config [files].mesh_mask is missing; the grid mesh path "
            "must be set per-run (no NEMO-version default)."
        ) from None


def get_obs_dir(config, override=None):
    """Base observations directory: ``override`` if given, else [files].obs_dir.

    ``override`` is a script's ``--obs-dir`` (None when not passed), letting the
    CLI redirect the directory while the filenames stay config-driven.
    """
    if override:
        return str(override)
    try:
        return _files(config)["obs_dir"]
    except KeyError:
        raise KeyError(
            "visualise_config [files].obs_dir is missing; set it per-run or pass "
            "--obs-dir."
        ) from None


def get_obs_filenames(config):
    """Mapping {logical_name: path-relative-to-obs_dir} from [files.obs]."""
    obs = _files(config).get("obs", {})
    if not obs:
        raise KeyError("visualise_config is missing the [files.obs] table.")
    return obs


def get_obs_path(config, key, override_dir=None):
    """Absolute path to observation file ``key`` (from [files.obs]).

    The directory is ``override_dir`` if given (a script's --obs-dir) else the
    config's [files].obs_dir; the filename is [files.obs][key]. Raises KeyError on
    an unknown key so a typo fails loudly rather than silently skipping a panel.
    """
    names = get_obs_filenames(config)
    try:
        rel = names[key]
    except KeyError:
        raise KeyError(
            f"unknown observation key '{key}'; [files.obs] defines: "
            f"{', '.join(sorted(names))}"
        ) from None
    return str(Path(get_obs_dir(config, override_dir)) / rel)
