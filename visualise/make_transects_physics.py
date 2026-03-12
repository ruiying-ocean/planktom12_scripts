#!/usr/bin/env python3
"""
Create zonal-mean depth-latitude section plots for physical variables.
Generates Atlantic and Pacific sections for temperature and salinity,
compared against WOA18 climatology (woa_orca_bil.nc on ORCA2 grid).

Each output figure: 2 rows (Temperature, Salinity) × 3 columns (Model | WOA18 | Difference)

Usage:
    python make_transects_physics.py <run_name> <year> [OPTIONS]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import interp1d

from logging_utils import print_warning


# Basin longitude masks (applied to the ORCA2 2-D lon grid)
BASINS = {
    'atlantic': {
        'label': 'Atlantic',
        'mask':  lambda lon: (lon >= -80) & (lon <= 20),
        'xlim':  (-75, 70),
    },
    'pacific': {
        'label': 'Pacific',
        'mask':  lambda lon: (lon >= 120) | (lon <= -70),
        'xlim':  (-60, 60),
    },
}

# (model_var, woa_var, display_name, units, section_levels, diff_levels)
PHYSICS_VARS = [
    {
        'model_var':   'votemper',
        'woa_var':     'temp',
        'name':        'Temperature',
        'units':       '°C',
        'levels':      np.arange(0, 30.5, 0.5),
        'diff_levels': np.arange(-2.0, 2.1, 0.2),
        'cmap':        'RdYlBu_r',
        'cmap_diff':   'RdBu_r',
    },
    {
        'model_var':   'vosaline',
        'woa_var':     'sal',
        'name':        'Salinity',
        'units':       'psu',
        'levels':      np.arange(33.5, 36.6, 0.1),
        'diff_levels': np.arange(-1.0, 1.05, 0.1),
        'cmap':        'RdYlBu_r',
        'cmap_diff':   'RdBu_r',
    },
]


def _zonal_mean(data3d, lon2d, basin_mask_fn):
    """
    Compute a zonal mean over a basin mask.

    Args:
        data3d: (depth, lat, lon) numpy masked array — ORCA2 2-D spatial grid
        lon2d:  (lat, lon) longitude array
        basin_mask_fn: callable returning a boolean mask for each row

    Returns:
        (depth, lat) masked array of zonal means
    """
    ndep, nlat = data3d.shape[:2]
    zm = np.ma.zeros((ndep, nlat))
    for j in range(nlat):
        cols = data3d[:, j, basin_mask_fn(lon2d[j, :])]
        zm[:, j] = cols.mean(axis=1) if cols.count() > 0 else np.ma.masked
    return zm


def plot_physics_sections(
    grid_t_file,
    obs_dir,
    output_dir,
    run_name: str,
    year: str,
):
    """
    Generate Atlantic and Pacific physical section plots (T and S).

    Reuses woa_orca_bil.nc (WOA18 bilinearly interpolated to ORCA2 grid)
    so no re-interpolation is needed for the difference panel.

    Args:
        grid_t_file: Path to ORCA2 grid_T NetCDF file (votemper, vosaline)
        obs_dir: Observations directory (must contain woa_orca_bil.nc)
        output_dir: Output directory for PNG files
        run_name: Model run name (used in filenames and titles)
        year: Year string (used in filenames and titles)
    """
    grid_t_file = Path(grid_t_file)
    obs_dir     = Path(obs_dir)
    output_dir  = Path(output_dir)

    woa_file = obs_dir / 'woa_orca_bil.nc'
    if not woa_file.exists():
        print_warning(f"woa_orca_bil.nc not found at {woa_file} — skipping physics sections")
        return

    # --- Load model grid_T (annual mean) ---
    ds_m   = xr.open_dataset(grid_t_file)
    lon_m  = ds_m['nav_lon'].values          # (j, i)
    deps_m = ds_m['deptht'].values           # (z,)
    lat1d  = ds_m['nav_lat'].values[:, ds_m['nav_lat'].shape[1] // 2]

    model_zm_cache = {}
    for v in PHYSICS_VARS:
        raw = ds_m[v['model_var']].mean(dim='time_counter').values
        raw = np.ma.masked_where(np.abs(raw) < 1e-6, raw)
        model_zm_cache[v['model_var']] = raw
    ds_m.close()

    # --- Load WOA on ORCA2 grid ---
    ds_w  = xr.open_dataset(woa_file, decode_times=False)
    # Detect WOA depth coordinate name
    woa_depth_dim = next(
        (d for d in ds_w.dims if 'depth' in d.lower()), None
    )
    deps_w = ds_w[woa_depth_dim].values if woa_depth_dim else None

    woa_zm_cache = {}
    for v in PHYSICS_VARS:
        if v['woa_var'] not in ds_w:
            print_warning(f"Variable '{v['woa_var']}' not in woa_orca_bil.nc — skipping {v['name']}")
            continue
        raw = ds_w[v['woa_var']].values
        # Drop time dim if present
        if raw.ndim == 4:
            raw = raw[0]
        # Use masked_invalid: xarray decodes fill values as NaN, so this is
        # sufficient without clobbering valid near-zero temperatures
        woa_zm_cache[v['woa_var']] = np.ma.masked_invalid(raw)
    ds_w.close()

    # --- One figure per basin ---
    for basin_key, basin_cfg in BASINS.items():
        mask_fn = basin_cfg['mask']

        fig, axes = plt.subplots(
            len(PHYSICS_VARS), 3,
            figsize=(18, 5 * len(PHYSICS_VARS)),
            constrained_layout=True
        )
        if len(PHYSICS_VARS) == 1:
            axes = axes[np.newaxis, :]

        for row, v in enumerate(PHYSICS_VARS):
            if v['model_var'] not in model_zm_cache or v['woa_var'] not in woa_zm_cache:
                continue

            ax_mod, ax_woa, ax_diff = axes[row]

            model_zm = _zonal_mean(model_zm_cache[v['model_var']], lon_m, mask_fn)
            woa_zm   = _zonal_mean(woa_zm_cache[v['woa_var']],     lon_m, mask_fn)

            # Interpolate WOA onto model depth levels for the difference panel
            # (WOA may have more depth levels than the model)
            if deps_w is not None and len(deps_w) != len(deps_m):
                woa_zm_on_m = np.ma.masked_all(model_zm.shape)
                for j in range(woa_zm.shape[1]):
                    col = woa_zm[:, j]
                    if col.count() > 0:
                        f = interp1d(deps_w, col.filled(np.nan),
                                     bounds_error=False, fill_value=np.nan)
                        woa_zm_on_m[:, j] = np.ma.masked_invalid(f(deps_m))
                    else:
                        woa_zm_on_m[:, j] = np.ma.masked
            else:
                woa_zm_on_m = woa_zm

            diff = model_zm - woa_zm_on_m
            deps_w_plot = deps_w if deps_w is not None else deps_m

            xlim = basin_cfg['xlim']

            # Model panel
            cf1 = ax_mod.contourf(
                lat1d, deps_m, model_zm,
                levels=v['levels'], cmap=v['cmap'], extend='both'
            )
            ax_mod.contour(
                lat1d, deps_m, model_zm,
                levels=v['levels'][::4], colors='k', linewidths=0.4
            )
            fig.colorbar(cf1, ax=ax_mod, label=v['units'])
            ax_mod.set_title(f"Model — {v['name']} ({year})")

            # WOA panel (plotted on its own depth grid)
            cf2 = ax_woa.contourf(
                lat1d, deps_w_plot, woa_zm,
                levels=v['levels'], cmap=v['cmap'], extend='both'
            )
            ax_woa.contour(
                lat1d, deps_w_plot, woa_zm,
                levels=v['levels'][::4], colors='k', linewidths=0.4
            )
            fig.colorbar(cf2, ax=ax_woa, label=v['units'])
            ax_woa.set_title(f"WOA18 — {v['name']}")

            # Difference panel
            cf3 = ax_diff.contourf(
                lat1d, deps_m, diff,
                levels=v['diff_levels'], cmap=v['cmap_diff'], extend='both'
            )
            ax_diff.contour(lat1d, deps_m, diff, levels=[0], colors='k', linewidths=1)
            fig.colorbar(cf3, ax=ax_diff, label=f"Model − WOA ({v['units']})")
            ax_diff.set_title(f"Difference — {v['name']}")

            for ax, dep_max in (
                (ax_mod,  deps_m[-1]),
                (ax_woa,  deps_w_plot[-1]),
                (ax_diff, deps_m[-1]),
            ):
                ax.set_ylim(dep_max, 0)
                ax.set_xlim(*xlim)
                ax.set_xlabel('Latitude (°N)')
                ax.set_ylabel('Depth (m)')

        fig.suptitle(
            f"{basin_cfg['label']} Physical Sections — Full Depth  "
            f"{run_name} ({year})",
            fontsize=13
        )

        out = output_dir / f"{run_name}_{year}_physics_section_{basin_key}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight',
                    pil_kwargs={'optimize': True, 'compress_level': 9})
        plt.close(fig)
        print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate physical (T/S) zonal-mean section plots'
    )
    parser.add_argument('run_name', help='Model run name')
    parser.add_argument('year',     help='Year to process (YYYY)')
    parser.add_argument('--model-run-dir', default='~/scratch/ModelRuns',
                        help='Directory containing model runs (default: %(default)s)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: <model-run-dir>/monitor/<run_name>)')
    parser.add_argument('--obs-dir',
                        default='/gpfs/home/vhf24tbu/Observations',
                        help='Observations directory containing woa_orca_bil.nc '
                             '(default: %(default)s)')

    args = parser.parse_args()

    model_run_dir = Path(args.model_run_dir).expanduser()
    run_dir       = model_run_dir / args.run_name
    output_dir    = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else model_run_dir / 'monitor' / args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str    = f"{args.year}0101_{args.year}1231"
    grid_t_file = run_dir / f"ORCA2_1m_{date_str}_grid_T.nc"

    if not grid_t_file.exists():
        print(f"Error: {grid_t_file} not found")
        sys.exit(1)

    plot_physics_sections(
        grid_t_file=grid_t_file,
        obs_dir=args.obs_dir,
        output_dir=output_dir,
        run_name=args.run_name,
        year=args.year,
    )


if __name__ == '__main__':
    main()
