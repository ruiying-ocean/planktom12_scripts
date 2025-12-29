#!/usr/bin/env python3
"""
Generate change maps for any PlankTom variable across ensemble models.

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

    # Mesozooplankton at specific depth (100m = level 10)
    python make_variable_change_maps.py <output_dir> --variable MES --depth-index 10

    # NPP (equivalent to make_npp_change_maps.py)
    python make_variable_change_maps.py <output_dir> --variable _PPINT

    # With custom periods
    python make_variable_change_maps.py <output_dir> --variable MES --hist-start 1990 --hist-end 2000
"""

import sys
import argparse
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
    '_NO3': 'ptrc_T',
    '_PO4': 'ptrc_T',
    '_Si': 'ptrc_T',
    '_Fer': 'ptrc_T',
    '_O2': 'ptrc_T',
    '_DIC': 'ptrc_T',
    '_ALK': 'ptrc_T',
    '_PHY': 'ptrc_T',
    '_ZOO': 'ptrc_T',

    # Integrated variables (computed from ptrc_T)
    '_MESINT': 'ptrc_T',
    '_PROINT': 'ptrc_T',
    '_BACINT': 'ptrc_T',
    '_PTEINT': 'ptrc_T',
    '_CRUINT': 'ptrc_T',
    '_GELINT': 'ptrc_T',
    '_PICINT': 'ptrc_T',
    '_FIXINT': 'ptrc_T',
    '_COCINT': 'ptrc_T',
    '_DIAINT': 'ptrc_T',
    '_MIXINT': 'ptrc_T',
    '_PHAINT': 'ptrc_T',
    '_PHYINT': 'ptrc_T',
    '_ZOOINT': 'ptrc_T',

    # grid_T variables
    'votemper': 'grid_T',
    'vosaline': 'grid_T',
    'tos': 'grid_T',
    'sos': 'grid_T',
    'mldr10_1': 'grid_T',
}

# Default depth indices for 3D variables
DEFAULT_DEPTH_INDEX = {
    '_O2': 17,      # ~300m
    'O2': 17,
    '_NO3': 0,      # surface
    '_TChl': 0,     # surface
}


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
    Discover model directories matching a pattern using Pathlib.

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

    # Find all matching directories
    matching_dirs = sorted(base_path.glob(pattern))

    # Filter to only directories
    model_dirs = [d for d in matching_dirs if d.is_dir()]

    models = []
    for model_path in model_dirs:
        models.append({
            'name': model_path.name,
            'desc': model_path.name,
            'model_dir': str(base_path)
        })

    return models


def get_file_pattern(variable, file_type=None):
    """Get the file pattern for a variable."""
    if file_type is None:
        file_type = VARIABLE_FILE_MAPPING.get(variable, 'ptrc_T')
    return f"ORCA2_1m_*_{file_type}.nc"


def load_period_average(model_dir, model_id, start_year, end_year, variable,
                        plotter, depth_index=None):
    """
    Load and compute multi-year average of a variable for a time period.

    Args:
        model_dir: Base directory for model output
        model_id: Model run name
        start_year: First year of period (inclusive)
        end_year: Last year of period (inclusive)
        variable: Variable name to load
        plotter: OceanMapPlotter instance
        depth_index: Depth level for 3D variables (None for 2D)

    Returns:
        xarray DataArray with time-averaged variable
    """
    run_dir = Path(model_dir) / model_id

    # Determine file type
    file_type = VARIABLE_FILE_MAPPING.get(variable, 'ptrc_T')

    yearly_data = []
    years_loaded = []

    for year in range(start_year, end_year + 1):
        nc_file = run_dir / f"ORCA2_1m_{year}0101_{year}1231_{file_type}.nc"

        if not nc_file.exists():
            print_warning(f"File not found: {nc_file}")
            continue

        try:
            # Load with preprocessing
            ds = plotter.load_data(str(nc_file), volume=plotter.volume)

            if variable not in ds:
                print_warning(f"{variable} not found in {nc_file}")
                ds.close()
                continue

            data = ds[variable]

            # Handle 3D variables - use top 200m mean (levels 0-12 ≈ 0-200m)
            if 'deptht' in data.dims:
                if depth_index is not None:
                    data = data.isel(deptht=depth_index)
                else:
                    # Default: mean over top 200m (levels 0-12)
                    data = data.isel(deptht=slice(0, 13)).mean(dim='deptht')

            # Annual mean for this year
            if 'time_counter' in data.dims:
                data = data.mean(dim='time_counter')

            yearly_data.append(data)
            years_loaded.append(year)
            ds.close()

        except Exception as e:
            print_warning(f"Error loading {year}: {e}")
            continue

    if not yearly_data:
        print_error(f"No data loaded for {model_id} ({start_year}-{end_year})")
        return None

    print_info(f"  Loaded {len(years_loaded)} years for {model_id}: {min(years_loaded)}-{max(years_loaded)}")

    # Stack and compute multi-year mean
    stacked = xr.concat(yearly_data, dim='year')
    period_mean = stacked.mean(dim='year')

    # Apply land mask
    period_mean = plotter.apply_mask(period_mean)

    return period_mean


def create_map_ax(fig, position, projection=ccrs.Robinson()):
    """Create a single map axis with cartopy features."""
    ax = fig.add_subplot(position, projection=projection)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.set_global()
    return ax


def get_variable_label(variable, depth_index=None, is_3d=False):
    """Get a human-readable label for a variable."""
    meta = get_variable_metadata(variable)
    label = meta.get('long_name', variable)
    units = meta.get('units', '')

    # Add depth info for 3D variables
    depth_str = ""
    if depth_index is not None:
        # Approximate depth from ORCA2 levels
        depths = [5, 15, 25, 36, 47, 59, 72, 86, 103, 122, 144, 169, 199, 233,
                  274, 322, 379, 446, 527, 623, 738, 876, 1042, 1242, 1483]
        if depth_index < len(depths):
            depth_str = f" @ {depths[depth_index]}m"
    elif is_3d:
        depth_str = " (0-200m mean)"

    return label, units, depth_str


def plot_variable_change_maps(models, output_dir, config, variable,
                               depth_index=None,
                               hist_start=2000, hist_end=2010,
                               fut_start=2090, fut_end=2100):
    """
    Create change maps for a variable for each model in the ensemble.

    Args:
        models: List of dicts with 'name', 'desc', 'model_dir'
        output_dir: Output directory
        config: Configuration dict
        variable: Variable name to plot
        depth_index: Depth level for 3D variables
        hist_start, hist_end: Historical period years
        fut_start, fut_end: Future period years
    """
    n_models = len(models)

    # Determine grid layout
    if n_models <= 3:
        n_cols = n_models
        n_rows = 1
    elif n_models <= 6:
        n_cols = 3
        n_rows = 2
    elif n_models <= 9:
        n_cols = 3
        n_rows = 3
    else:
        n_cols = 4
        n_rows = (n_models + 3) // 4

    # Get DPI and format from config
    dpi = config.get("figure", {}).get("dpi", 300) if config else 300
    fmt = config.get("figure", {}).get("format", "png") if config else "png"

    # Create OceanMapPlotter for data preprocessing
    plotter = OceanMapPlotter()

    projection = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    # Load navigation coordinates from first available model
    nav_lon, nav_lat = None, None
    file_type = VARIABLE_FILE_MAPPING.get(variable, 'ptrc_T')

    for model in models:
        run_dir = Path(model['model_dir']) / model['name']
        data_files = sorted(run_dir.glob(f"ORCA2_1m_*_{file_type}.nc"))
        if data_files:
            ds = xr.open_dataset(data_files[0])
            nav_lon = ds.nav_lon if 'nav_lon' in ds else ds.lon
            nav_lat = ds.nav_lat if 'nav_lat' in ds else ds.lat
            ds.close()
            break

    if nav_lon is None:
        print_error("Could not load navigation from any model files")
        return

    # Store all model data for consistent color scaling
    all_changes = []
    model_data = {}

    print_header(f"Loading {variable} data for {n_models} models...")

    for model in models:
        model_name = model['name']
        model_dir = model['model_dir']

        print_info(f"Processing {model_name}...")

        # Load historical period
        hist_data = load_period_average(
            model_dir, model_name, hist_start, hist_end,
            variable, plotter, depth_index
        )
        if hist_data is None:
            model_data[model_name] = None
            continue

        # Load future period
        fut_data = load_period_average(
            model_dir, model_name, fut_start, fut_end,
            variable, plotter, depth_index
        )
        if fut_data is None:
            model_data[model_name] = None
            continue

        # Calculate change
        change = fut_data - hist_data
        model_data[model_name] = {
            'historical': hist_data,
            'future': fut_data,
            'change': change
        }

        # Collect valid change values for color scale
        valid_vals = change.values[~np.isnan(change.values)]
        all_changes.extend(valid_vals.flatten())

    if not all_changes:
        print_error("No valid data to plot")
        return

    # Determine symmetric color scale using percentiles
    all_changes = np.array(all_changes)
    vmax = np.percentile(np.abs(all_changes), 95)

    # Determine if variable is 3D (needs depth handling)
    file_type = VARIABLE_FILE_MAPPING.get(variable, 'ptrc_T')
    is_3d = file_type == 'ptrc_T' and not variable.endswith('INT')

    # Get variable metadata
    var_label, var_units, depth_str = get_variable_label(variable, depth_index, is_3d)

    # Create figure
    subplot_width = 5
    subplot_height = 3
    fig = plt.figure(figsize=(subplot_width * n_cols, subplot_height * n_rows + 1),
                     constrained_layout=True)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    # Plot each model
    im = None
    for idx, model in enumerate(models):
        model_name = model['name']

        row = idx // n_cols
        col = idx % n_cols

        ax = create_map_ax(fig, gs[row, col], projection)

        if model_name in model_data and model_data[model_name] is not None:
            change = model_data[model_name]['change']

            im = ax.pcolormesh(
                nav_lon, nav_lat, change,
                transform=data_crs,
                cmap='RdBu_r',
                vmin=-vmax, vmax=vmax,
                shading='auto'
            )

        ax.set_title(model_name, fontsize=10, fontweight='bold')

    # Add shared colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
        cbar.set_label(f'Δ{var_label}{depth_str} ({var_units})\n{fut_start}-{fut_end} minus {hist_start}-{hist_end}',
                       fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    # Add main title
    fig.suptitle(f'{var_label}{depth_str} Change\n({hist_start}-{hist_end} → {fut_start}-{fut_end})',
                 fontsize=14, fontweight='bold', y=1.02)

    # Save figure
    var_safe = variable.replace('_', '').lower()
    depth_suffix = f"_z{depth_index}" if depth_index is not None else ""
    output_file = output_dir / f"{var_safe}{depth_suffix}_change_{hist_start}_{hist_end}_to_{fut_start}_{fut_end}.{fmt}"
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print_success(f"Created {output_file}")
    plt.close(fig)

    # Also create individual period maps
    plot_period_maps(models, model_data, nav_lon, nav_lat, output_dir, config,
                     variable, depth_index, hist_start, hist_end, fut_start, fut_end)


def plot_period_maps(models, model_data, nav_lon, nav_lat, output_dir, config,
                     variable, depth_index, hist_start, hist_end, fut_start, fut_end):
    """
    Create maps showing absolute values for each period (historical and future).
    """
    n_models = len(models)

    # Grid layout
    if n_models <= 3:
        n_cols = n_models
        n_rows = 1
    elif n_models <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 4
        n_rows = (n_models + 3) // 4

    dpi = config.get("figure", {}).get("dpi", 300) if config else 300
    fmt = config.get("figure", {}).get("format", "png") if config else "png"

    projection = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    # Get variable metadata
    meta = get_variable_metadata(variable)
    vmax = meta.get('vmax', None)
    vmin = meta.get('vmin', 0)
    cmap = meta.get('cmap', 'viridis')

    # Determine if variable is 3D
    file_type = VARIABLE_FILE_MAPPING.get(variable, 'ptrc_T')
    is_3d = file_type == 'ptrc_T' and not variable.endswith('INT')
    var_label, var_units, depth_str = get_variable_label(variable, depth_index, is_3d)

    # If no vmax defined, compute from data
    if vmax is None:
        all_vals = []
        for model_name, data in model_data.items():
            if data is not None:
                for period in ['historical', 'future']:
                    vals = data[period].values[~np.isnan(data[period].values)]
                    all_vals.extend(vals.flatten())
        if all_vals:
            vmax = np.percentile(np.abs(all_vals), 95)
        else:
            vmax = 1

    for period_name, period_key, start_yr, end_yr in [
        ('historical', 'historical', hist_start, hist_end),
        ('future', 'future', fut_start, fut_end)
    ]:
        subplot_width = 5
        subplot_height = 3
        fig = plt.figure(figsize=(subplot_width * n_cols, subplot_height * n_rows + 1),
                         constrained_layout=True)
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

        im = None
        for idx, model in enumerate(models):
            model_name = model['name']

            row = idx // n_cols
            col = idx % n_cols

            ax = create_map_ax(fig, gs[row, col], projection)

            if model_name in model_data and model_data[model_name] is not None:
                data = model_data[model_name][period_key]

                im = ax.pcolormesh(
                    nav_lon, nav_lat, data,
                    transform=data_crs,
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    shading='auto'
                )

            ax.set_title(model_name, fontsize=10, fontweight='bold')

        if im is not None:
            cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='max')
            cbar.set_label(f'{var_label}{depth_str} ({var_units})', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        fig.suptitle(f'{var_label}{depth_str} ({start_yr}-{end_yr} Average)',
                     fontsize=14, fontweight='bold', y=1.02)

        var_safe = variable.replace('_', '').lower()
        depth_suffix = f"_z{depth_index}" if depth_index is not None else ""
        output_file = output_dir / f"{var_safe}{depth_suffix}_{period_name}_{start_yr}_{end_yr}.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print_success(f"Created {output_file}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate change maps for any PlankTom variable across ensemble models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mesozooplankton biomass (0-200m mean by default)
  python make_variable_change_maps.py ./output --variable MES

  # Mesozooplankton at specific depth (100m = level 10)
  python make_variable_change_maps.py ./output --variable MES --depth-index 10

  # NPP change maps
  python make_variable_change_maps.py ./output --variable _PPINT

  # Oxygen at 300m
  python make_variable_change_maps.py ./output --variable _O2 --depth-index 17

Available variables:
  Phytoplankton: PIC, FIX, COC, DIA, MIX, PHA (0-200m mean)
  Zooplankton:   MES, PRO, BAC, PTE, CRU, GEL (0-200m mean)
  Nutrients:     _NO3, _PO4, _Si, _Fer, _O2
  Production:    _PPINT (NPP), _EXP (export), _TChl, _eratio, _Teff
"""
    )
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for plots')
    parser.add_argument('--variable', '-v', type=str, default='MES',
                        help='Variable to plot (default: MES)')
    parser.add_argument('--depth-index', '-z', type=int, default=None,
                        help='Depth level index for 3D variables (default: 0-200m mean; 0=surface, 10=~100m, 17=~300m)')
    parser.add_argument('--base-dir', type=Path,
                        default=Path(os.path.expanduser("~/scratch/ModelRuns")),
                        help='Base directory containing model runs (default: ~/scratch/ModelRuns)')
    parser.add_argument('--pattern', type=str, default="TOM12_RY_JRA*",
                        help='Glob pattern to match model directories (default: TOM12_RY_JRA*)')
    parser.add_argument('--hist-start', type=int, default=2000,
                        help='Start year for historical period (default: 2000)')
    parser.add_argument('--hist-end', type=int, default=2010,
                        help='End year for historical period (default: 2010)')
    parser.add_argument('--fut-start', type=int, default=2090,
                        help='Start year for future period (default: 2090)')
    parser.add_argument('--fut-end', type=int, default=2100,
                        help='End year for future period (default: 2100)')

    args = parser.parse_args()

    output_dir = args.output_dir

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config()

    # Discover models using Pathlib glob
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

    plot_variable_change_maps(
        models, output_dir, config,
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
