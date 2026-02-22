#!/usr/bin/env python3
"""
Compute spatial R², RMSE, and bias between model surface fields and
gridded observations for chlorophyll and nutrients on the ORCA2 grid.

Prints a compact terminal table — no plotting dependencies required.

Usage:
    python make_scorecard.py ERA3 --year 2024
    python make_scorecard.py ERA3 --year 2024 --monthly
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


# ── Variable definitions ────────────────────────────────────────────────
VARIABLES = [
    {
        "name": "TChl",
        "model_var": "TChl",
        "model_file": "diad",
        "obs_file": "OC-CCI/climatology/OC-CCI_climatology_orca2.nc",
        "obs_var": "chlor_a",
        "model_factor": 1e6,        # raw diad units → mg/m³ (OC-CCI unit)
        "unit": "mg/m³",
        "surface_only": False,      # TChl has deptht; extract surface
        "obs_has_months": True,
    },
    {
        "name": "NO3",
        "model_var": "NO3",
        "model_file": "ptrc",
        "obs_file": "woa_orca_bil.nc",
        "obs_var": "no3",
        "model_factor": 1e6,        # mol/L → µmol/L
        "unit": "µmol/L",
        "surface_only": False,
        "obs_has_months": False,
    },
    {
        "name": "PO4",
        "model_var": "PO4",
        "model_file": "ptrc",
        "obs_file": "woa_orca_bil.nc",
        "obs_var": "po4",
        "model_factor": 1e6 / 122,   # mol-C/L → µmol-P/L (Redfield C:P=122)
        "unit": "µmol/L",
        "surface_only": False,
        "obs_has_months": False,
    },
    {
        "name": "Si",
        "model_var": "Si",
        "model_file": "ptrc",
        "obs_file": "woa_orca_bil.nc",
        "obs_var": "si",
        "model_factor": 1e6,
        "unit": "µmol/L",
        "surface_only": False,
        "obs_has_months": False,
    },
    {
        "name": "Fer",
        "model_var": "Fer",
        "model_file": "ptrc",
        "obs_file": "Huang2022_orca.nc",
        "obs_var": "fe",
        "model_factor": 1e9,        # mol/L → nmol/L
        "unit": "nmol/L",
        "surface_only": False,
        "obs_has_months": False,
    },
]

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ── Metrics ─────────────────────────────────────────────────────────────
def compute_metrics(model_vals, obs_vals):
    """Return (R², RMSE, bias, M-score) for paired 1-D arrays."""
    mask = np.isfinite(model_vals) & np.isfinite(obs_vals)
    m = model_vals[mask]
    o = obs_vals[mask]
    n = len(m)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(m, o)[0, 1]
    r2 = r ** 2
    rmse = np.sqrt(np.mean((m - o) ** 2))
    bias = np.mean(m - o)
    # M-score (Watterson, 1996)
    mse = np.mean((m - o) ** 2)
    denom = m.var() + o.var() + (m.mean() - o.mean()) ** 2
    mscore = (2.0 / np.pi) * np.arcsin(1.0 - mse / denom) if denom > 0 else np.nan
    return r2, rmse, bias, mscore


# ── Data loading helpers ────────────────────────────────────────────────
def _ensure_2d(arr):
    """Collapse leading dimensions until the array is 2-D (y, x)."""
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _get_surface(da):
    """Select surface level from a DataArray (handles various dim names)."""
    if "deptht" in da.dims:
        da = da.isel(deptht=0)
    for zdim in ("depth", "z"):
        if zdim in da.dims:
            da = da.isel({zdim: 0})
    return da


def load_model_surface(ds, var_name, factor):
    """Extract annual-mean surface field from a model dataset."""
    da = _get_surface(ds[var_name])
    if "time_counter" in da.dims:
        da = da.mean(dim="time_counter")
    return _ensure_2d(da.squeeze().values * factor)


def load_model_surface_monthly(ds, var_name, factor):
    """Extract 12 monthly surface fields from a model dataset.

    Returns list of 12 2-D arrays (one per month).
    """
    da = _get_surface(ds[var_name])
    out = []
    for t in range(da.sizes.get("time_counter", 1)):
        month_da = da.isel(time_counter=t) if "time_counter" in da.dims else da
        out.append(_ensure_2d(month_da.squeeze().values * factor))
    return out


def load_obs_surface(obs_path, obs_var):
    """Load the surface layer of an observational dataset (annual mean)."""
    ds = xr.open_dataset(str(obs_path), decode_times=False)
    da = ds[obs_var]
    for tdim in ("time_counter", "time", "t"):
        if tdim in da.dims:
            da = da.mean(dim=tdim)
    da = _get_surface(da)
    vals = _ensure_2d(da.squeeze().values.astype(np.float64))
    ds.close()
    return vals


def load_obs_surface_monthly(obs_path, obs_var):
    """Load 12 monthly surface fields from an obs dataset.

    Returns list of 12 2-D arrays.
    """
    ds = xr.open_dataset(str(obs_path), decode_times=False)
    da = ds[obs_var]
    da = _get_surface(da)
    # Find time dimension
    tdim = None
    for candidate in ("time_counter", "time", "t"):
        if candidate in da.dims:
            tdim = candidate
            break
    out = []
    for t in range(da.sizes.get(tdim, 1)):
        month_da = da.isel({tdim: t}) if tdim else da
        out.append(_ensure_2d(month_da.squeeze().values.astype(np.float64)))
    ds.close()
    return out


def _compare(mod, obs):
    """Flatten two 2-D arrays, mask, and return metrics."""
    if mod.shape != obs.shape:
        return None
    mod_f = mod.ravel()
    obs_f = obs.ravel()
    valid = (
        np.isfinite(mod_f) & np.isfinite(obs_f)
        & (mod_f != 0) & (obs_f != 0)
    )
    return compute_metrics(mod_f[valid], obs_f[valid])


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Surface model-vs-observations scorecard (R², RMSE, bias)"
    )
    parser.add_argument("run_name", help="Model run name (e.g. ERA3)")
    parser.add_argument("--year", required=True, help="Year to process (YYYY)")
    parser.add_argument(
        "--monthly", action="store_true",
        help="Show per-month statistics for variables with monthly obs (TChl)",
    )
    parser.add_argument(
        "--model-run-dir",
        default="~/scratch/ModelRuns",
        help="Base directory containing model runs (default: %(default)s)",
    )
    parser.add_argument(
        "--obs-dir",
        default="/gpfs/home/vhf24tbu/Observations",
        help="Directory containing observational data files (default: %(default)s)",
    )
    args = parser.parse_args()

    model_run_dir = Path(args.model_run_dir).expanduser()
    obs_dir = Path(args.obs_dir).expanduser()
    run_dir = model_run_dir / args.run_name

    date_str = f"{args.year}0101_{args.year}1231"
    ptrc_file = run_dir / f"ORCA2_1m_{date_str}_ptrc_T.nc"
    diad_file = run_dir / f"ORCA2_1m_{date_str}_diad_T.nc"

    for fpath, label in [(ptrc_file, "ptrc_T"), (diad_file, "diad_T")]:
        if not fpath.exists():
            print(f"Error: {label} file not found: {fpath}", file=sys.stderr)
            sys.exit(1)

    ptrc_ds = xr.open_dataset(str(ptrc_file), decode_times=False)
    diad_ds = xr.open_dataset(str(diad_file), decode_times=False)
    model_datasets = {"ptrc": ptrc_ds, "diad": diad_ds}

    # ── Compute & collect rows ──────────────────────────────────────────
    rows = []  # list of (label, r2, rmse, bias, mscore, unit)
    for var in VARIABLES:
        obs_path = obs_dir / var["obs_file"]
        if not obs_path.exists():
            print(f"Warning: obs file not found for {var['name']}: {obs_path}",
                  file=sys.stderr)
            continue

        ds = model_datasets[var["model_file"]]
        if var["model_var"] not in ds:
            print(f"Warning: {var['model_var']} not found in model {var['model_file']}_T",
                  file=sys.stderr)
            continue

        if args.monthly and var.get("obs_has_months"):
            # Pool all months into a single comparison
            mod_months = load_model_surface_monthly(
                ds, var["model_var"], var["model_factor"]
            )
            obs_months = load_obs_surface_monthly(obs_path, var["obs_var"])
            n_months = min(len(mod_months), len(obs_months))
            mod_all, obs_all = [], []
            for m in range(n_months):
                if mod_months[m].shape != obs_months[m].shape:
                    print(f"Warning: shape mismatch for {var['name']} month {m+1}",
                          file=sys.stderr)
                    continue
                mod_f = mod_months[m].ravel()
                obs_f = obs_months[m].ravel()
                valid = (
                    np.isfinite(mod_f) & np.isfinite(obs_f)
                    & (mod_f != 0) & (obs_f != 0)
                )
                mod_all.append(mod_f[valid])
                obs_all.append(obs_f[valid])
            if mod_all:
                mod_cat = np.concatenate(mod_all)
                obs_cat = np.concatenate(obs_all)
                r2, rmse, bias, mscore = compute_metrics(mod_cat, obs_cat)
                rows.append((var["name"], r2, rmse, bias, mscore, var["unit"]))
        else:
            # Annual mean comparison
            mod = load_model_surface(ds, var["model_var"], var["model_factor"])
            obs = load_obs_surface(obs_path, var["obs_var"])
            result = _compare(mod, obs)
            if result is None:
                print(f"Warning: shape mismatch for {var['name']}: "
                      f"model {mod.shape} vs obs {obs.shape}", file=sys.stderr)
                continue
            r2, rmse, bias, mscore = result
            rows.append((var["name"], r2, rmse, bias, mscore, var["unit"]))

    ptrc_ds.close()
    diad_ds.close()

    # ── Print table ─────────────────────────────────────────────────────
    mode = "Monthly" if args.monthly else "Annual"
    sep = "\u2500" * 70
    print(f"\nSurface Model-Obs Statistics: {args.run_name} (year {args.year}, {mode})")
    print(sep)
    print(f"{'Variable':<12} {'R²':>8} {'RMSE':>10} {'Bias':>10} {'M':>8}  Unit")
    print(sep)
    for label, r2, rmse, bias, mscore, unit in rows:
        r2_s = f"{r2:.2f}" if np.isfinite(r2) else "  N/A"
        rmse_s = f"{rmse:.3f}" if np.isfinite(rmse) else "     N/A"
        bias_s = f"{bias:+.3f}" if np.isfinite(bias) else "     N/A"
        m_s = f"{mscore:.2f}" if np.isfinite(mscore) else "  N/A"
        print(f"{label:<12} {r2_s:>8} {rmse_s:>10} {bias_s:>10} {m_s:>8}  {unit}")
    print(sep)


if __name__ == "__main__":
    main()
