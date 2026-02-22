#!/usr/bin/env python3
"""
Compute spatial R², RMSE, and bias between model surface fields and
gridded observations for chlorophyll and nutrients on the ORCA2 grid.

Prints a compact terminal table — no plotting dependencies required.

Usage:
    python make_scorecard.py ERA3 --year 2024
    python make_scorecard.py ERA3 --year 2024 --model-run-dir ~/scratch/ModelRuns --obs-dir ~/Observations
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
        "model_factor": 1.0,
        "unit": "µg/L",
        "surface_only": True,       # diad TChl is already 2-D
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
    },
]


# ── Metrics ─────────────────────────────────────────────────────────────
def compute_metrics(model_vals, obs_vals):
    """Return (R², RMSE, bias, N) for paired 1-D arrays."""
    mask = np.isfinite(model_vals) & np.isfinite(obs_vals)
    m = model_vals[mask]
    o = obs_vals[mask]
    n = len(m)
    if n < 2:
        return np.nan, np.nan, np.nan, n
    r = np.corrcoef(m, o)[0, 1]
    r2 = r ** 2
    rmse = np.sqrt(np.mean((m - o) ** 2))
    bias = np.mean(m - o)
    return r2, rmse, bias, n


# ── Data loading helpers ────────────────────────────────────────────────
def _ensure_2d(arr):
    """Collapse leading dimensions until the array is 2-D (y, x)."""
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def load_model_surface(ds, var_name, surface_only, factor):
    """Extract annual-mean surface field from a model dataset."""
    da = ds[var_name]
    # Time average
    if "time_counter" in da.dims:
        da = da.mean(dim="time_counter")
    # Surface extraction (skip for 2-D fields like TChl)
    if not surface_only and "deptht" in da.dims:
        da = da.isel(deptht=0)
    da = da.squeeze()
    return _ensure_2d(da.values * factor)


def load_obs_surface(obs_path, obs_var):
    """Load the surface layer of an observational dataset."""
    ds = xr.open_dataset(str(obs_path), decode_times=False)
    da = ds[obs_var]
    # Average over time if present
    for tdim in ("time_counter", "time", "t"):
        if tdim in da.dims:
            da = da.mean(dim=tdim)
    # Take surface if 3-D
    for zdim in ("deptht", "depth", "z"):
        if zdim in da.dims:
            da = da.isel({zdim: 0})
    da = da.squeeze()
    vals = _ensure_2d(da.values.astype(np.float64))
    ds.close()
    return vals


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Surface model-vs-observations scorecard (R², RMSE, bias)"
    )
    parser.add_argument("run_name", help="Model run name (e.g. ERA3)")
    parser.add_argument("--year", required=True, help="Year to process (YYYY)")
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

    # Check model files
    for fpath, label in [(ptrc_file, "ptrc_T"), (diad_file, "diad_T")]:
        if not fpath.exists():
            print(f"Error: {label} file not found: {fpath}", file=sys.stderr)
            sys.exit(1)

    # Open model datasets once
    ptrc_ds = xr.open_dataset(str(ptrc_file), decode_times=False)
    diad_ds = xr.open_dataset(str(diad_file), decode_times=False)
    model_datasets = {"ptrc": ptrc_ds, "diad": diad_ds}

    # ── Compute & collect rows ──────────────────────────────────────────
    rows = []
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

        mod = load_model_surface(
            ds, var["model_var"], var["surface_only"], var["model_factor"]
        )
        obs = load_obs_surface(obs_path, var["obs_var"])

        if mod.shape != obs.shape:
            print(f"Warning: shape mismatch for {var['name']}: "
                  f"model {mod.shape} vs obs {obs.shape}", file=sys.stderr)
            continue

        # Flatten to 1-D and build common valid-ocean mask
        mod_f = mod.ravel()
        obs_f = obs.ravel()
        valid = (
            np.isfinite(mod_f) & np.isfinite(obs_f)
            & (mod_f != 0) & (obs_f != 0)
        )
        r2, rmse, bias, n = compute_metrics(mod_f[valid], obs_f[valid])
        rows.append((var["name"], r2, rmse, bias, n, var["unit"]))

    ptrc_ds.close()
    diad_ds.close()

    # ── Print table ─────────────────────────────────────────────────────
    sep = "\u2500" * 60
    print(f"\nSurface Model-Obs Statistics: {args.run_name} (year {args.year})")
    print(sep)
    print(f"{'Variable':<12} {'R²':>8} {'RMSE':>10} {'Bias':>10} {'N':>8}  Unit")
    print(sep)
    for name, r2, rmse, bias, n, unit in rows:
        r2_s = f"{r2:.2f}" if np.isfinite(r2) else "  N/A"
        rmse_s = f"{rmse:.3f}" if np.isfinite(rmse) else "     N/A"
        bias_s = f"{bias:+.3f}" if np.isfinite(bias) else "     N/A"
        print(f"{name:<12} {r2_s:>8} {rmse_s:>10} {bias_s:>10} {n:>8}  {unit}")
    print(sep)


if __name__ == "__main__":
    main()
