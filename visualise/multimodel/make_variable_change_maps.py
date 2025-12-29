#!/usr/bin/env python3
"""
Generate change maps for any PlankTom variable across ensemble models.

Memory-optimized version that processes models sequentially and uses
streaming computation to avoid loading all data into memory at once.

Computes the difference between two time periods:
- Historical: 2000-2010 average (default)
- Future: 2090-2100 average (default)
- Change: Future - Historical

Supports variables from:
- diad_T files: _PPINT (NPP), _EXP (export), _TChl (chlorophyll), etc.
- ptrc_T files: MES (mesozooplankton), PFTs, nutrients at specific depths
- grid_T files: temperature, salinity, MLD

Usage:
    # Mesozooplankton biomass (0-200m mean, default)
    python make_variable_change_maps.py <output_dir> --variable MES

    # NPP (replaces make_npp_change_maps.py)
    python make_variable_change_maps.py <output_dir> --variable _PPINT

    # Mesozooplankton at specific depth (100m = level 10)
    python make_variable_change_maps.py <output_dir> --variable MES --depth-index 10

    # With custom periods
    python make_variable_change_maps.py <output_dir> --variable MES --hist-start 1990 --hist-end 2000
"""

import sys
import argparse
import gc
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# Import map utilities from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from map_utils import OceanMapPlotter, get_variable_metadata, PHYTOS, ZOOS
from logging_utils import print_header, print_info, print_warning, print_error, print_success

# Import configuration
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# Variable configuration: which file type contains each variable
VARIABLE_FILE_MAPPING = {
    # diad_T variables (2D, already integrated/surface)
    '_PPINT': 'diad_T',
    '_EXP': 'diad_T',
    '_TChl': 'diad_T',
    '_eratio': 'diad_T',
    '_Teff': 'diad_T',
    '_SP': 'diad_T',
    'Cflx': 'diad_T',
    'dpco2': 'diad_T',
    'PPINT': 'diad_T',
    'PPT': 'diad_T',
    'EXP': 'diad_T',
    'TChl': 'diad_T',

    # ptrc_T variables (3D, need depth selection or integration)
    'MES': 'ptrc_T',
    'PRO': 'ptrc_T',
    'BAC': 'ptrc_T',
    'PTE': 'ptrc_T',
    'CRU': 'ptrc_T',
    'GEL': 'ptrc_T',
    'PIC': 'ptrc_T',
    'FIX': 'ptrc_T',
    'COC': 'ptrc_T',
    'DIA': 'ptrc_T',
    'MIX': 'ptrc_T',
    'PHA': 'ptrc_T',
    'NO3': 'ptrc_T',
    'PO4': 'ptrc_T',
    'Si': 'ptrc_T',
    'Fer': 'ptrc_T',
    'O2': 'ptrc_T',
    'DIC': 'ptrc_T',
    'Alkalini': 'ptrc_T',

    # grid_T variables
    'votemper': 'grid_T',
    'vosaline': 'grid_T',
    'tos': 'grid_T',
    'sos': 'grid_T',
    'mldr10_1': 'grid_T',
}

# Composite variables that need to be computed from multiple fields
COMPOSITE_VARIABLES = {
    '_PHY': ['PIC', 'FIX', 'COC', 'DIA', 'MIX', 'PHA'],
    '_ZOO': ['BAC', 'PRO', 'MES', 'PTE', 'CRU', 'GEL'],
}

# ORCA2 depth levels (approximate depths in meters)
ORCA2_DEPTHS = [4.999938e+00, 1.500029e+01, 2.500176e+01, 3.500541e+01, 4.501332e+01,
       5.502950e+01, 6.506181e+01, 7.512551e+01, 8.525037e+01, 9.549429e+01,
       1.059699e+02, 1.168962e+02, 1.286979e+02, 1.421953e+02, 1.589606e+02,
       1.819628e+02, 2.166479e+02, 2.724767e+02, 3.643030e+02, 5.115348e+02,
       7.322009e+02, 1.033217e+03, 1.405698e+03, 1.830885e+03, 2.289768e+03,
       2.768242e+03, 3.257479e+03, 3.752442e+03, 4.250401e+03, 4.749913e+03,
       5.250227e+03]


def load_config():
    """Load visualise_config.toml"""
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "visualise_config.toml"

    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return None


def discover_models(base_dir, pattern="TOM12_RY_JRA*"):
    """
    Discover model directories matching a pattern.

    Args:
        base_dir: Base directory containing model runs
        pattern: Glob pattern to match model directories

    Returns:
        List of model dicts with 'name' and 'model_dir'
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print_error(f"Base directory not found: {base_dir}")
        return []

    matching_dirs = sorted(base_path.glob(pattern))
    model_dirs = [d for d in matching_dirs if d.is_dir()]

    models = []
    for model_path in model_dirs:
        models.append({
            'name': model_path.name,
            'model_dir': str(base_path)
        })

    return models


def load_variable_year(nc_file, variable, depth_index=None, depth_range=(0, 13)):
    """
    Memory-efficient loading of a single variable for one year.

    Args:
        nc_file: Path to NetCDF file
        variable: Variable name to load
        depth_index: Specific depth level (None for depth averaging)
        depth_range: Tuple (start, end) for depth averaging (default 0-200m)

    Returns:
        2D numpy array (y, x) with annual mean, or None on error
    """
    try:
        # Open with minimal loading - only coordinates and the variable we need
        with xr.open_dataset(nc_file, decode_times=False) as ds:
            # Handle composite variables
            if variable in COMPOSITE_VARIABLES:
                components = COMPOSITE_VARIABLES[variable]
                data = None
                for comp in components:
                    if comp in ds:
                        comp_data = ds[comp].values
                        if data is None:
                            data = comp_data.copy()
                        else:
                            data += comp_data
                    else:
                        return None
                # Convert to DataArray for consistent handling
                if 'deptht' in ds[components[0]].dims:
                    data_da = xr.DataArray(
                        data,
                        dims=ds[components[0]].dims,
                        coords={k: ds[components[0]].coords[k] for k in ds[components[0]].coords}
                    )
                else:
                    data_da = xr.DataArray(data, dims=ds[components[0]].dims)
            elif variable not in ds:
                return None
            else:
                data_da = ds[variable]

            # Handle 3D variables - reduce depth dimension first to save memory
            if 'deptht' in data_da.dims:
                if depth_index is not None:
                    # Single depth level
                    data_da = data_da.isel(deptht=depth_index)
                else:
                    # Mean over depth range (default 0-200m = levels 0-12)
                    data_da = data_da.isel(deptht=slice(*depth_range)).mean(dim='deptht')

            # Compute annual mean
            if 'time_counter' in data_da.dims:
                data_da = data_da.mean(dim='time_counter')

            # Return as numpy array to free xarray overhead
            return data_da.values

    except Exception as e:
        print_warning(f"Error loading {variable} from {nc_file}: {e}")
        return None


def load_period_mean_streaming(model_dir, model_id, start_year, end_year,
                                variable, file_type, depth_index=None,
                                land_mask=None):
    """
    Memory-efficient period averaging using streaming computation.

    Loads one year at a time and computes running mean to avoid
    storing all years in memory.

    Args:
        model_dir: Base directory for model output
        model_id: Model run name
        start_year: First year of period (inclusive)
        end_year: Last year of period (inclusive)
        variable: Variable name to load
        file_type: File type (diad_T, ptrc_T, grid_T)
        depth_index: Depth level for 3D variables
        land_mask: Optional 2D boolean mask for land

    Returns:
        2D numpy array with period mean, or None on failure
    """
    run_dir = Path(model_dir) / model_id

    running_sum = None
    n_years = 0

    for year in range(start_year, end_year + 1):
        nc_file = run_dir / f"ORCA2_1m_{year}0101_{year}1231_{file_type}.nc"

        if not nc_file.exists():
            continue

        year_data = load_variable_year(nc_file, variable, depth_index)

        if year_data is None:
            continue

        if running_sum is None:
            running_sum = year_data.astype(np.float64)
        else:
            running_sum += year_data

        n_years += 1

        # Explicitly free memory
        del year_data
        gc.collect()

    if n_years == 0:
        print_warning(f"No data loaded for {model_id} ({start_year}-{end_year})")
        return None

    period_mean = running_sum / n_years

    # Apply land mask if provided
    if land_mask is not None:
        period_mean = np.where(land_mask, period_mean, np.nan)

    print_info(f"  {model_id}: loaded {n_years} years ({start_year}-{end_year})")

    return period_mean


def get_nav_coords_and_mask(models, file_type):
    """
    Load navigation coordinates and land mask from first available model.

    Returns:
        Tuple of (nav_lon, nav_lat, land_mask) numpy arrays
    """
    for model in models:
        run_dir = Path(model['model_dir']) / model['name']
        data_files = sorted(run_dir.glob(f"ORCA2_1m_*_{file_type}.nc"))

        if data_files:
            with xr.open_dataset(data_files[0]) as ds:
                nav_lon = ds['nav_lon'].values if 'nav_lon' in ds else ds['lon'].values
                nav_lat = ds['nav_lat'].values if 'nav_lat' in ds else ds['lat'].values

                # Create land mask from first variable with data
                for var in ds.data_vars:
                    if ds[var].dims[-2:] == ('y', 'x'):
                        sample = ds[var].isel(time_counter=0)
                        if 'deptht' in sample.dims:
                            sample = sample.isel(deptht=0)
                        land_mask = ~np.isnan(sample.values)
                        return nav_lon, nav_lat, land_mask

    return None, None, None


def create_map_ax(fig, position, projection=ccrs.Robinson()):
    """Create a single map axis with cartopy features."""
    ax = fig.add_subplot(position, projection=projection)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.set_global()
    return ax


def get_grid_layout(n_models):
    """Determine optimal grid layout for n models."""
    if n_models <= 3:
        return 1, n_models
    elif n_models <= 6:
        return 2, 3
    elif n_models <= 9:
        return 3, 3
    else:
        n_cols = 4
        n_rows = (n_models + 3) // 4
        return n_rows, n_cols


def get_variable_info(variable, depth_index=None):
    """
    Get display information for a variable.

    Returns:
        Tuple of (label, units, depth_string, is_3d)
    """
    meta = get_variable_metadata(variable)
    label = meta.get('long_name', variable)
    units = meta.get('units', '')

    file_type = VARIABLE_FILE_MAPPING.get(variable, 'ptrc_T')
    is_3d = file_type == 'ptrc_T'

    if depth_index is not None and depth_index < len(ORCA2_DEPTHS):
        depth_str = f" @ {ORCA2_DEPTHS[depth_index]}m"
    elif is_3d:
        depth_str = " (0-200m mean)"
    else:
        depth_str = ""

    return label, units, depth_str, is_3d


def plot_change_maps(models, output_dir, config, variable,
                     depth_index=None,
                     hist_start=2000, hist_end=2010,
                     fut_start=2090, fut_end=2100):
    """
    Create change maps with memory-efficient processing.

    Processes models in two passes:
    1. First pass: compute changes and collect statistics for color scaling
    2. Second pass: create the plot using stored change arrays
    """
    n_models = len(models)
    n_rows, n_cols = get_grid_layout(n_models)

    # Get config values
    dpi = config.get("figure", {}).get("dpi", 300) if config else 300
    fmt = config.get("figure", {}).get("format", "png") if config else "png"

    # Determine file type
    file_type = VARIABLE_FILE_MAPPING.get(variable, 'ptrc_T')

    # Load coordinates and mask
    nav_lon, nav_lat, land_mask = get_nav_coords_and_mask(models, file_type)
    if nav_lon is None:
        print_error("Could not load navigation from any model files")
        return

    # Get variable display info
    var_label, var_units, depth_str, is_3d = get_variable_info(variable, depth_index)

    print_header(f"Processing {variable} for {n_models} models...")

    # First pass: compute changes and collect stats
    # Store only the change arrays (not hist/fut separately)
    model_changes = {}
    all_change_vals = []

    for model in models:
        model_name = model['name']
        model_dir = model['model_dir']

        print_info(f"Processing {model_name}...")

        # Load historical period
        hist_mean = load_period_mean_streaming(
            model_dir, model_name, hist_start, hist_end,
            variable, file_type, depth_index, land_mask
        )

        if hist_mean is None:
            model_changes[model_name] = None
            continue

        # Load future period
        fut_mean = load_period_mean_streaming(
            model_dir, model_name, fut_start, fut_end,
            variable, file_type, depth_index, land_mask
        )

        if fut_mean is None:
            model_changes[model_name] = None
            del hist_mean
            gc.collect()
            continue

        # Compute change
        change = fut_mean - hist_mean
        model_changes[model_name] = change

        # Collect valid values for color scaling
        valid = change[~np.isnan(change)]
        all_change_vals.extend(valid.flatten().tolist())

        # Free intermediate arrays
        del hist_mean, fut_mean
        gc.collect()

    if not all_change_vals:
        print_error("No valid data to plot")
        return

    # Compute symmetric color scale
    all_change_vals = np.array(all_change_vals)
    vmax = np.percentile(np.abs(all_change_vals), 95)
    del all_change_vals
    gc.collect()

    # Second pass: create the plot
    projection = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(5 * n_cols, 3 * n_rows + 1), constrained_layout=True)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    im = None
    for idx, model in enumerate(models):
        model_name = model['name']
        row, col = idx // n_cols, idx % n_cols

        ax = create_map_ax(fig, gs[row, col], projection)

        if model_name in model_changes and model_changes[model_name] is not None:
            change = model_changes[model_name]

            im = ax.pcolormesh(
                nav_lon, nav_lat, change,
                transform=data_crs,
                cmap='RdBu_r',
                vmin=-vmax, vmax=vmax,
                shading='auto'
            )

        ax.set_title(model_name, fontsize=10, fontweight='bold')

    # Add colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
        cbar.set_label(
            f'Δ{var_label}{depth_str} ({var_units})\n{fut_start}-{fut_end} minus {hist_start}-{hist_end}',
            fontsize=12
        )
        cbar.ax.tick_params(labelsize=10)

    # Title
    fig.suptitle(
        f'{var_label}{depth_str} Change\n({hist_start}-{hist_end} → {fut_start}-{fut_end})',
        fontsize=14, fontweight='bold', y=1.02
    )

    # Save
    var_safe = variable.replace('_', '').lower()
    depth_suffix = f"_z{depth_index}" if depth_index is not None else ""
    output_file = output_dir / f"{var_safe}{depth_suffix}_change_{hist_start}_{hist_end}_to_{fut_start}_{fut_end}.{fmt}"
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print_success(f"Created {output_file}")
    plt.close(fig)

    # Clean up
    del model_changes
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description='Generate change maps for any PlankTom variable across ensemble models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # NPP change maps (replaces make_npp_change_maps.py)
  python make_variable_change_maps.py ./output --variable _PPINT

  # Mesozooplankton biomass (0-200m mean by default)
  python make_variable_change_maps.py ./output --variable MES

  # Mesozooplankton at specific depth (100m = level 10)
  python make_variable_change_maps.py ./output --variable MES --depth-index 10

  # Oxygen at 300m
  python make_variable_change_maps.py ./output --variable O2 --depth-index 17

Available variables:
  Production:    _PPINT (NPP), _EXP (export), _TChl, PPT
  Phytoplankton: PIC, FIX, COC, DIA, MIX, PHA (0-200m mean)
  Zooplankton:   MES, PRO, BAC, PTE, CRU, GEL (0-200m mean)
  Aggregates:    _PHY (total phyto), _ZOO (total zoo)
  Nutrients:     NO3, PO4, Si, Fer, O2
"""
    )
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for plots')
    parser.add_argument('--variable', '-v', type=str, default='_PPINT',
                        help='Variable to plot (default: _PPINT)')
    parser.add_argument('--depth-index', '-z', type=int, default=None,
                        help='Depth level index for 3D variables (default: 0-200m mean)')
    parser.add_argument('--base-dir', type=Path,
                        default=Path(os.path.expanduser("~/scratch/ModelRuns")),
                        help='Base directory containing model runs')
    parser.add_argument('--pattern', type=str, default="TOM12_RY_JRA*",
                        help='Glob pattern to match model directories')
    parser.add_argument('--hist-start', type=int, default=2000,
                        help='Start year for historical period (default: 2000)')
    parser.add_argument('--hist-end', type=int, default=2010,
                        help='End year for historical period (default: 2010)')
    parser.add_argument('--fut-start', type=int, default=2090,
                        help='Start year for future period (default: 2090)')
    parser.add_argument('--fut-end', type=int, default=2100,
                        help='End year for future period (default: 2100)')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config()

    # Discover models
    models = discover_models(args.base_dir, args.pattern)

    if not models:
        print_error(f"No models found matching pattern '{args.pattern}' in {args.base_dir}")
        return 1

    print_header(f"Found {len(models)} models matching '{args.pattern}':")
    for model in models:
        print_info(f"  - {model['name']}")

    print_info(f"Variable: {args.variable}")
    if args.depth_index is not None:
        print_info(f"Depth index: {args.depth_index}")
    print_info(f"Historical period: {args.hist_start}-{args.hist_end}")
    print_info(f"Future period: {args.fut_start}-{args.fut_end}")

    plot_change_maps(
        models, args.output_dir, config,
        variable=args.variable,
        depth_index=args.depth_index,
        hist_start=args.hist_start,
        hist_end=args.hist_end,
        fut_start=args.fut_start,
        fut_end=args.fut_end
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
