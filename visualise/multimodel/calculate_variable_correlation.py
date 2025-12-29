#!/usr/bin/env python3
"""
Calculate correlation between two variables across ensemble models for emergent constraints.

This script computes spatial means/integrals for two variables across all models
and calculates cross-model correlations. This is useful for identifying emergent
constraints where a measurable predictor (X) correlates with a target (Y).

Two modes:
1. Single-value mode (default): Compute area-weighted spatial means, then correlate
   across models. Output: scatter plot + CSV.
2. Spatial map mode (--spatial-map): Compute cross-model correlation at each grid
   cell. Output: map showing where the emergent constraint relationship is strongest.

Supports latitude constraints for regional analysis (single-value mode only):
- Global: all latitudes
- Tropical: 30°S to 30°N
- High latitudes: poleward of 45°
- Custom latitude ranges

Usage:
    # Global correlation between historical NPP and future MES change
    python calculate_variable_correlation.py ./output --var-x _PPINT --var-y MES

    # Tropical correlation
    python calculate_variable_correlation.py ./output --var-x _PPINT --var-y MES --lat-min -30 --lat-max 30

    # High latitude Southern Ocean
    python calculate_variable_correlation.py ./output --var-x _PPINT --var-y MES --lat-min -90 --lat-max -45

    # Spatial map of cross-model correlations
    python calculate_variable_correlation.py ./output --var-x _PPINT --var-y MES --spatial-map

    # X as historical mean, Y as future-historical change
    python calculate_variable_correlation.py ./output --var-x _PPINT --var-y MES --x-mode hist --y-mode change
"""

import sys
import argparse
import gc
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
import os

# Import map utilities from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from map_utils import get_variable_metadata
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

            # Handle 3D variables
            if 'deptht' in data_da.dims:
                if depth_index is not None:
                    data_da = data_da.isel(deptht=depth_index)
                else:
                    data_da = data_da.isel(deptht=slice(*depth_range)).mean(dim='deptht')

            # Compute annual mean
            if 'time_counter' in data_da.dims:
                data_da = data_da.mean(dim='time_counter')

            return data_da.values

    except Exception as e:
        print_warning(f"Error loading {variable} from {nc_file}: {e}")
        return None


def load_period_mean_streaming(model_dir, model_id, start_year, end_year,
                                variable, file_type, depth_index=None,
                                land_mask=None):
    """
    Memory-efficient period averaging using streaming computation.

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
        del year_data
        gc.collect()

    if n_years == 0:
        return None

    period_mean = running_sum / n_years

    if land_mask is not None:
        period_mean = np.where(land_mask, period_mean, np.nan)

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

                for var in ds.data_vars:
                    if ds[var].dims[-2:] == ('y', 'x'):
                        sample = ds[var].isel(time_counter=0)
                        if 'deptht' in sample.dims:
                            sample = sample.isel(deptht=0)
                        land_mask = ~np.isnan(sample.values)
                        return nav_lon, nav_lat, land_mask

    return None, None, None


def load_area_weights(mask_path="/gpfs/data/greenocean/software/resources/breakdown/basin_mask.nc"):
    """Load area weights from basin mask file."""
    try:
        with xr.open_dataset(mask_path) as ds:
            area_key = 'AREA' if 'AREA' in ds else 'area'
            if area_key in ds:
                return ds[area_key].values
    except Exception:
        pass
    return None


def compute_spatial_mean(data_2d, nav_lat, area_weights=None, lat_min=-90, lat_max=90):
    """
    Compute area-weighted spatial mean within latitude bounds.

    Args:
        data_2d: 2D array (y, x)
        nav_lat: 2D array of latitudes
        area_weights: 2D array of cell areas (optional, uses cos(lat) if None)
        lat_min: Minimum latitude
        lat_max: Maximum latitude

    Returns:
        Scalar area-weighted mean
    """
    # Create latitude mask
    lat_mask = (nav_lat >= lat_min) & (nav_lat <= lat_max)

    # Combine with valid data mask
    valid_mask = ~np.isnan(data_2d) & lat_mask

    if not np.any(valid_mask):
        return np.nan

    # Use area weights or approximate with cos(lat)
    if area_weights is not None:
        weights = area_weights.copy()
    else:
        weights = np.cos(np.deg2rad(nav_lat))

    # Apply masks
    masked_data = np.where(valid_mask, data_2d, 0)
    masked_weights = np.where(valid_mask, weights, 0)

    total_weight = np.sum(masked_weights)
    if total_weight == 0:
        return np.nan

    return np.sum(masked_data * masked_weights) / total_weight


def get_variable_info(variable, depth_index=None):
    """Get display information for a variable."""
    meta = get_variable_metadata(variable)
    label = meta.get('long_name', variable)
    units = meta.get('units', '')

    file_type = VARIABLE_FILE_MAPPING.get(variable, 'ptrc_T')
    is_3d = file_type == 'ptrc_T'

    if depth_index is not None and depth_index < len(ORCA2_DEPTHS):
        depth_str = f" @ {ORCA2_DEPTHS[depth_index]}m"
    elif is_3d:
        depth_str = " (0-200m)"
    else:
        depth_str = ""

    return label, units, depth_str, is_3d


def calculate_correlation(models, config,
                          var_x, var_y,
                          x_mode='change', y_mode='change',
                          depth_x=None, depth_y=None,
                          lat_min=-90, lat_max=90,
                          hist_start=2000, hist_end=2010,
                          fut_start=2090, fut_end=2100):
    """
    Calculate cross-model correlation between two variables.

    Args:
        models: List of model dicts
        config: Configuration dict
        var_x: X-axis variable name
        var_y: Y-axis variable name
        x_mode: 'hist' (historical mean), 'fut' (future mean), or 'change'
        y_mode: 'hist', 'fut', or 'change'
        depth_x: Depth index for var_x (None for depth mean)
        depth_y: Depth index for var_y (None for depth mean)
        lat_min: Minimum latitude for spatial averaging
        lat_max: Maximum latitude for spatial averaging
        hist_start, hist_end: Historical period
        fut_start, fut_end: Future period

    Returns:
        Dict with correlation results and data
    """
    # Determine file types
    file_type_x = VARIABLE_FILE_MAPPING.get(var_x, 'ptrc_T')
    file_type_y = VARIABLE_FILE_MAPPING.get(var_y, 'ptrc_T')

    # Use the first file type for coordinates (they should be the same grid)
    nav_lon, nav_lat, land_mask = get_nav_coords_and_mask(models, file_type_x)
    if nav_lon is None:
        print_error("Could not load navigation coordinates")
        return None

    # Load area weights
    area_weights = load_area_weights()

    x_values = []
    y_values = []
    model_names = []

    for model in models:
        model_name = model['name']
        model_dir = model['model_dir']

        print_info(f"Processing {model_name}...")

        # Load X variable
        if x_mode in ['hist', 'change']:
            x_hist = load_period_mean_streaming(
                model_dir, model_name, hist_start, hist_end,
                var_x, file_type_x, depth_x, land_mask
            )
        else:
            x_hist = None

        if x_mode in ['fut', 'change']:
            x_fut = load_period_mean_streaming(
                model_dir, model_name, fut_start, fut_end,
                var_x, file_type_x, depth_x, land_mask
            )
        else:
            x_fut = None

        # Compute X value based on mode
        if x_mode == 'hist':
            x_field = x_hist
        elif x_mode == 'fut':
            x_field = x_fut
        else:  # change
            if x_hist is not None and x_fut is not None:
                x_field = x_fut - x_hist
            else:
                x_field = None

        # Load Y variable
        if y_mode in ['hist', 'change']:
            y_hist = load_period_mean_streaming(
                model_dir, model_name, hist_start, hist_end,
                var_y, file_type_y, depth_y, land_mask
            )
        else:
            y_hist = None

        if y_mode in ['fut', 'change']:
            y_fut = load_period_mean_streaming(
                model_dir, model_name, fut_start, fut_end,
                var_y, file_type_y, depth_y, land_mask
            )
        else:
            y_fut = None

        # Compute Y value based on mode
        if y_mode == 'hist':
            y_field = y_hist
        elif y_mode == 'fut':
            y_field = y_fut
        else:  # change
            if y_hist is not None and y_fut is not None:
                y_field = y_fut - y_hist
            else:
                y_field = None

        # Compute spatial means
        if x_field is not None and y_field is not None:
            x_mean = compute_spatial_mean(x_field, nav_lat, area_weights, lat_min, lat_max)
            y_mean = compute_spatial_mean(y_field, nav_lat, area_weights, lat_min, lat_max)

            if not np.isnan(x_mean) and not np.isnan(y_mean):
                x_values.append(x_mean)
                y_values.append(y_mean)
                model_names.append(model_name)
                print_info(f"  {model_name}: X={x_mean:.4g}, Y={y_mean:.4g}")
            else:
                print_warning(f"  {model_name}: NaN values, skipping")
        else:
            print_warning(f"  {model_name}: Could not load data, skipping")

        # Clean up
        del x_hist, x_fut, y_hist, y_fut, x_field, y_field
        gc.collect()

    if len(x_values) < 3:
        print_error(f"Not enough valid models ({len(x_values)} < 3) for correlation")
        return None

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Calculate correlation
    r, p_value = stats.pearsonr(x_values, y_values)

    # Linear regression
    slope, intercept, r_value, p_val, std_err = stats.linregress(x_values, y_values)

    return {
        'x_values': x_values,
        'y_values': y_values,
        'model_names': model_names,
        'r': r,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err,
        'n_models': len(x_values)
    }


def plot_correlation(results, output_dir, config,
                     var_x, var_y,
                     x_mode, y_mode,
                     depth_x, depth_y,
                     lat_min, lat_max,
                     hist_start, hist_end,
                     fut_start, fut_end):
    """Create scatter plot with regression line."""

    dpi = config.get("figure", {}).get("dpi", 300) if config else 300
    fmt = config.get("figure", {}).get("format", "png") if config else "png"

    # Get variable info
    x_label, x_units, x_depth, _ = get_variable_info(var_x, depth_x)
    y_label, y_units, y_depth, _ = get_variable_info(var_y, depth_y)

    # Build axis labels with mode info
    mode_labels = {
        'hist': f'({hist_start}-{hist_end})',
        'fut': f'({fut_start}-{fut_end})',
        'change': f'Δ ({fut_start}-{fut_end} - {hist_start}-{hist_end})'
    }

    x_axis_label = f"{x_label}{x_depth} {mode_labels[x_mode]}"
    if x_units:
        x_axis_label += f" [{x_units}]"

    y_axis_label = f"{y_label}{y_depth} {mode_labels[y_mode]}"
    if y_units:
        y_axis_label += f" [{y_units}]"

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    x_vals = results['x_values']
    y_vals = results['y_values']

    # Scatter plot
    ax.scatter(x_vals, y_vals, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Regression line
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_line = results['slope'] * x_line + results['intercept']
    ax.plot(x_line, y_line, 'r-', linewidth=2, label='Linear fit')

    # Add model labels
    for i, name in enumerate(results['model_names']):
        # Shorten model name for display
        short_name = name.replace('TOM12_RY_JRA_', '').replace('TOM12_RY_', '')
        ax.annotate(short_name, (x_vals[i], y_vals[i]),
                    fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')

    # Statistics text
    stats_text = (f"r = {results['r']:.3f}\n"
                  f"p = {results['p_value']:.3e}\n"
                  f"n = {results['n_models']}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Labels and title
    ax.set_xlabel(x_axis_label, fontsize=11)
    ax.set_ylabel(y_axis_label, fontsize=11)

    # Title with latitude info
    lat_str = f"{lat_min}°-{lat_max}°" if lat_min != -90 or lat_max != 90 else "Global"
    ax.set_title(f"Cross-Model Correlation ({lat_str})", fontsize=12, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    plt.tight_layout()

    # Save
    var_x_safe = var_x.replace('_', '').lower()
    var_y_safe = var_y.replace('_', '').lower()
    lat_suffix = f"_lat{lat_min}to{lat_max}" if lat_min != -90 or lat_max != 90 else ""
    output_file = output_dir / f"correlation_{var_x_safe}_{x_mode}_vs_{var_y_safe}_{y_mode}{lat_suffix}.{fmt}"

    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print_success(f"Saved plot: {output_file}")
    plt.close(fig)

    # Also save data to CSV
    csv_file = output_file.with_suffix('.csv')
    with open(csv_file, 'w') as f:
        f.write(f"# Correlation: {var_x} ({x_mode}) vs {var_y} ({y_mode})\n")
        f.write(f"# Latitude range: {lat_min} to {lat_max}\n")
        f.write(f"# r = {results['r']:.4f}, p = {results['p_value']:.4e}\n")
        f.write(f"# slope = {results['slope']:.4e}, intercept = {results['intercept']:.4e}\n")
        f.write("model,x_value,y_value\n")
        for name, x, y in zip(results['model_names'], results['x_values'], results['y_values']):
            f.write(f"{name},{x},{y}\n")
    print_success(f"Saved data: {csv_file}")


def calculate_spatial_correlation_map(models, var_x, var_y,
                                       x_mode='change', y_mode='change',
                                       depth_x=None, depth_y=None,
                                       hist_start=2000, hist_end=2010,
                                       fut_start=2090, fut_end=2100,
                                       min_models=3):
    """
    Memory-efficient spatial correlation map.

    For each year, loads data from all models and computes cross-model
    correlation at each grid cell. Accumulates running statistics.

    Args:
        models: List of model dicts
        var_x: X-axis variable name
        var_y: Y-axis variable name
        x_mode: 'hist', 'fut', or 'change'
        y_mode: 'hist', 'fut', or 'change'
        depth_x, depth_y: Depth indices (None for 0-200m mean)
        hist_start, hist_end: Historical period
        fut_start, fut_end: Future period
        min_models: Minimum number of valid models per grid cell

    Returns:
        Dict with correlation_map, nav_lon, nav_lat, n_valid_models
    """
    file_type_x = VARIABLE_FILE_MAPPING.get(var_x, 'ptrc_T')
    file_type_y = VARIABLE_FILE_MAPPING.get(var_y, 'ptrc_T')

    # Get coordinates and mask
    nav_lon, nav_lat, land_mask = get_nav_coords_and_mask(models, file_type_x)
    if nav_lon is None:
        print_error("Could not load navigation coordinates")
        return None

    n_models = len(models)
    grid_shape = nav_lon.shape

    # Determine years to process based on modes
    if x_mode == 'change' or y_mode == 'change':
        # Need both periods
        years_hist = list(range(hist_start, hist_end + 1))
        years_fut = list(range(fut_start, fut_end + 1))
    elif x_mode == 'hist' or y_mode == 'hist':
        years_hist = list(range(hist_start, hist_end + 1))
        years_fut = []
    else:  # both are 'fut'
        years_hist = []
        years_fut = list(range(fut_start, fut_end + 1))

    print_info(f"Processing {len(years_hist)} historical years, {len(years_fut)} future years")

    # Accumulators for Pearson correlation using running sums
    # r = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_x2 - sum_x^2)*(n*sum_y2 - sum_y^2))
    sum_x = np.zeros(grid_shape, dtype=np.float64)
    sum_y = np.zeros(grid_shape, dtype=np.float64)
    sum_x2 = np.zeros(grid_shape, dtype=np.float64)
    sum_y2 = np.zeros(grid_shape, dtype=np.float64)
    sum_xy = np.zeros(grid_shape, dtype=np.float64)
    n_valid = np.zeros(grid_shape, dtype=np.int32)

    # For change mode, we need to compute period means first per model
    # Then correlate across models
    # This requires loading all model data for each period

    print_header("Loading model data...")

    # Storage for period means per model
    x_fields = {}  # model_name -> 2D array
    y_fields = {}  # model_name -> 2D array

    for model in models:
        model_name = model['name']
        model_dir = model['model_dir']
        print_info(f"Processing {model_name}...")

        # Load X variable based on mode
        if x_mode == 'hist':
            x_field = load_period_mean_streaming(
                model_dir, model_name, hist_start, hist_end,
                var_x, file_type_x, depth_x, land_mask
            )
        elif x_mode == 'fut':
            x_field = load_period_mean_streaming(
                model_dir, model_name, fut_start, fut_end,
                var_x, file_type_x, depth_x, land_mask
            )
        else:  # change
            x_hist = load_period_mean_streaming(
                model_dir, model_name, hist_start, hist_end,
                var_x, file_type_x, depth_x, land_mask
            )
            x_fut = load_period_mean_streaming(
                model_dir, model_name, fut_start, fut_end,
                var_x, file_type_x, depth_x, land_mask
            )
            if x_hist is not None and x_fut is not None:
                x_field = x_fut - x_hist
            else:
                x_field = None
            del x_hist, x_fut

        # Load Y variable based on mode
        if y_mode == 'hist':
            y_field = load_period_mean_streaming(
                model_dir, model_name, hist_start, hist_end,
                var_y, file_type_y, depth_y, land_mask
            )
        elif y_mode == 'fut':
            y_field = load_period_mean_streaming(
                model_dir, model_name, fut_start, fut_end,
                var_y, file_type_y, depth_y, land_mask
            )
        else:  # change
            y_hist = load_period_mean_streaming(
                model_dir, model_name, hist_start, hist_end,
                var_y, file_type_y, depth_y, land_mask
            )
            y_fut = load_period_mean_streaming(
                model_dir, model_name, fut_start, fut_end,
                var_y, file_type_y, depth_y, land_mask
            )
            if y_hist is not None and y_fut is not None:
                y_field = y_fut - y_hist
            else:
                y_field = None
            del y_hist, y_fut

        if x_field is not None and y_field is not None:
            x_fields[model_name] = x_field
            y_fields[model_name] = y_field
        else:
            print_warning(f"  {model_name}: Could not load data, skipping")

        gc.collect()

    valid_models = list(x_fields.keys())
    n_valid_models = len(valid_models)
    print_info(f"Loaded data from {n_valid_models} models")

    if n_valid_models < min_models:
        print_error(f"Not enough valid models ({n_valid_models} < {min_models})")
        return None

    # Compute correlation at each grid cell across models
    print_header("Computing spatial correlation map...")

    # Stack into arrays: (n_models, y, x)
    x_stack = np.stack([x_fields[m] for m in valid_models], axis=0)
    y_stack = np.stack([y_fields[m] for m in valid_models], axis=0)

    # Free memory
    del x_fields, y_fields
    gc.collect()

    # Compute Pearson correlation at each grid cell
    # Using vectorized formula
    n = n_valid_models

    # Count valid (non-NaN) models at each grid cell
    valid_mask = ~np.isnan(x_stack) & ~np.isnan(y_stack)
    n_valid_per_cell = np.sum(valid_mask, axis=0)

    # Replace NaN with 0 for computation, will mask later
    x_stack = np.where(valid_mask, x_stack, 0)
    y_stack = np.where(valid_mask, y_stack, 0)

    # Compute sums only over valid entries
    sum_x = np.sum(x_stack, axis=0)
    sum_y = np.sum(y_stack, axis=0)
    sum_x2 = np.sum(x_stack**2, axis=0)
    sum_y2 = np.sum(y_stack**2, axis=0)
    sum_xy = np.sum(x_stack * y_stack, axis=0)

    # Pearson correlation formula
    n_arr = n_valid_per_cell.astype(np.float64)
    numerator = n_arr * sum_xy - sum_x * sum_y
    denom_x = n_arr * sum_x2 - sum_x**2
    denom_y = n_arr * sum_y2 - sum_y**2

    # Handle edge cases
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = np.sqrt(denom_x * denom_y)
        correlation_map = np.where(denominator > 0, numerator / denominator, np.nan)

    # Mask cells with too few valid models
    correlation_map = np.where(n_valid_per_cell >= min_models, correlation_map, np.nan)

    # Clean up
    del x_stack, y_stack
    gc.collect()

    return {
        'correlation_map': correlation_map,
        'nav_lon': nav_lon,
        'nav_lat': nav_lat,
        'n_valid_models': n_valid_per_cell,
        'model_names': valid_models
    }


def plot_correlation_map(results, output_dir, config,
                         var_x, var_y,
                         x_mode, y_mode,
                         depth_x, depth_y,
                         hist_start, hist_end,
                         fut_start, fut_end):
    """Create map of cross-model correlations."""

    dpi = config.get("figure", {}).get("dpi", 300) if config else 300
    fmt = config.get("figure", {}).get("format", "png") if config else "png"

    correlation_map = results['correlation_map']
    nav_lon = results['nav_lon']
    nav_lat = results['nav_lat']

    # Get variable info for labels
    x_label, x_units, x_depth, _ = get_variable_info(var_x, depth_x)
    y_label, y_units, y_depth, _ = get_variable_info(var_y, depth_y)

    # Mode labels
    mode_labels = {
        'hist': f'{hist_start}-{hist_end}',
        'fut': f'{fut_start}-{fut_end}',
        'change': f'Δ({fut_start}-{fut_end} - {hist_start}-{hist_end})'
    }

    # Create figure
    projection = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.set_global()

    # Plot correlation map with diverging colormap
    im = ax.pcolormesh(
        nav_lon, nav_lat, correlation_map,
        transform=data_crs,
        cmap='RdBu_r',
        vmin=-1, vmax=1,
        shading='auto'
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05,
                        shrink=0.8, extend='neither')
    cbar.set_label('Pearson r (cross-model)', fontsize=11)

    # Title
    title = (f"Cross-Model Correlation\n"
             f"{x_label}{x_depth} ({mode_labels[x_mode]}) vs "
             f"{y_label}{y_depth} ({mode_labels[y_mode]})")
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save PNG
    var_x_safe = var_x.replace('_', '').lower()
    var_y_safe = var_y.replace('_', '').lower()
    output_file = output_dir / f"correlation_map_{var_x_safe}_{x_mode}_vs_{var_y_safe}_{y_mode}.{fmt}"
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print_success(f"Saved plot: {output_file}")
    plt.close(fig)

    # Save to NetCDF for further analysis
    nc_file = output_file.with_suffix('.nc')
    ds = xr.Dataset({
        'correlation': (['y', 'x'], correlation_map),
        'n_valid_models': (['y', 'x'], results['n_valid_models']),
    }, coords={
        'nav_lon': (['y', 'x'], nav_lon),
        'nav_lat': (['y', 'x'], nav_lat),
    })
    ds.attrs['var_x'] = var_x
    ds.attrs['var_y'] = var_y
    ds.attrs['x_mode'] = x_mode
    ds.attrs['y_mode'] = y_mode
    ds.attrs['hist_period'] = f'{hist_start}-{hist_end}'
    ds.attrs['fut_period'] = f'{fut_start}-{fut_end}'
    ds.attrs['models'] = ','.join(results['model_names'])
    ds.to_netcdf(nc_file)
    print_success(f"Saved NetCDF: {nc_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate cross-model correlation between two variables for emergent constraints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Global correlation: historical NPP vs future MES change
  python calculate_variable_correlation.py ./output --var-x _PPINT --var-y MES

  # Tropical region only
  python calculate_variable_correlation.py ./output --var-x _PPINT --var-y MES --lat-min -30 --lat-max 30

  # Southern Ocean (high latitudes)
  python calculate_variable_correlation.py ./output --var-x _PPINT --var-y MES --lat-min -90 --lat-max -45

  # Spatial map of cross-model correlations
  python calculate_variable_correlation.py ./output --var-x _PPINT --var-y MES --spatial-map

  # Both variables as changes
  python calculate_variable_correlation.py ./output --var-x _PPINT --var-y MES --x-mode change --y-mode change

Modes:
  hist   : Historical period mean
  fut    : Future period mean
  change : Future minus historical (default for both X and Y)
"""
    )
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for plots and data')
    parser.add_argument('--var-x', '-x', type=str, default='_PPINT',
                        help='X-axis variable (predictor, default: _PPINT)')
    parser.add_argument('--var-y', '-y', type=str, default='MES',
                        help='Y-axis variable (target, default: MES)')
    parser.add_argument('--x-mode', type=str, choices=['hist', 'fut', 'change'], default='change',
                        help='X variable mode: hist, fut, or change (default: change)')
    parser.add_argument('--y-mode', type=str, choices=['hist', 'fut', 'change'], default='change',
                        help='Y variable mode: hist, fut, or change (default: change)')
    parser.add_argument('--depth-x', type=int, default=None,
                        help='Depth index for X variable (default: 0-200m mean for 3D)')
    parser.add_argument('--depth-y', type=int, default=None,
                        help='Depth index for Y variable (default: 0-200m mean for 3D)')
    parser.add_argument('--lat-min', type=float, default=-90,
                        help='Minimum latitude for spatial averaging (default: -90)')
    parser.add_argument('--lat-max', type=float, default=90,
                        help='Maximum latitude for spatial averaging (default: 90)')
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
    parser.add_argument('--spatial-map', action='store_true',
                        help='Output spatial map of cross-model correlations instead of single value')

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

    print_header(f"Emergent Constraint Analysis")
    print_info(f"Found {len(models)} models matching '{args.pattern}'")
    print_info(f"X variable: {args.var_x} ({args.x_mode})")
    print_info(f"Y variable: {args.var_y} ({args.y_mode})")
    print_info(f"Historical period: {args.hist_start}-{args.hist_end}")
    print_info(f"Future period: {args.fut_start}-{args.fut_end}")
    if args.spatial_map:
        print_info("Mode: Spatial correlation map")
    else:
        print_info(f"Latitude range: {args.lat_min}° to {args.lat_max}°")

    if args.spatial_map:
        # Spatial correlation map mode
        results = calculate_spatial_correlation_map(
            models,
            var_x=args.var_x,
            var_y=args.var_y,
            x_mode=args.x_mode,
            y_mode=args.y_mode,
            depth_x=args.depth_x,
            depth_y=args.depth_y,
            hist_start=args.hist_start,
            hist_end=args.hist_end,
            fut_start=args.fut_start,
            fut_end=args.fut_end
        )

        if results is None:
            return 1

        print_header("Spatial Correlation Map Results")
        corr_map = results['correlation_map']
        valid_corrs = corr_map[~np.isnan(corr_map)]
        print_info(f"Mean correlation: {np.mean(valid_corrs):.4f}")
        print_info(f"Correlation range: {np.min(valid_corrs):.4f} to {np.max(valid_corrs):.4f}")
        print_info(f"Number of models: {len(results['model_names'])}")

        # Plot map
        plot_correlation_map(
            results, args.output_dir, config,
            var_x=args.var_x,
            var_y=args.var_y,
            x_mode=args.x_mode,
            y_mode=args.y_mode,
            depth_x=args.depth_x,
            depth_y=args.depth_y,
            hist_start=args.hist_start,
            hist_end=args.hist_end,
            fut_start=args.fut_start,
            fut_end=args.fut_end
        )
    else:
        # Original single-value correlation mode
        results = calculate_correlation(
            models, config,
            var_x=args.var_x,
            var_y=args.var_y,
            x_mode=args.x_mode,
            y_mode=args.y_mode,
            depth_x=args.depth_x,
            depth_y=args.depth_y,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            hist_start=args.hist_start,
            hist_end=args.hist_end,
            fut_start=args.fut_start,
            fut_end=args.fut_end
        )

        if results is None:
            return 1

        print_header("Correlation Results")
        print_info(f"Pearson r = {results['r']:.4f}")
        print_info(f"p-value = {results['p_value']:.4e}")
        print_info(f"Number of models = {results['n_models']}")

        # Plot results
        plot_correlation(
            results, args.output_dir, config,
            var_x=args.var_x,
            var_y=args.var_y,
            x_mode=args.x_mode,
            y_mode=args.y_mode,
            depth_x=args.depth_x,
            depth_y=args.depth_y,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            hist_start=args.hist_start,
            hist_end=args.hist_end,
            fut_start=args.fut_start,
            fut_end=args.fut_end
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
