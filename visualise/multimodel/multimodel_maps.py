#!/usr/bin/env python3
"""
Generate multi-model spatial comparison maps in grid format.
Creates single PNG files with models in columns, variables in rows.

For 2 models: columns = Model A, Model B, Anomaly (A-B)
For N models: columns = Model 1, Model 2, ..., Model N

Usage:
    python multimodel_maps.py <models_csv> <output_dir>
"""

import sys
import csv
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import map utilities from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from map_utils import OceanMapPlotter, get_variable_metadata

# Import configuration
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config():
    """Load visualise_config.toml"""
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "visualise_config.toml"

    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return None


def load_model_data(basedir, run_name, year, var_name, plotter):
    """
    Load annual mean surface data for a variable using OceanMapPlotter.

    Args:
        basedir: Base directory for model output
        run_name: Model run name
        year: Year to load
        var_name: Variable name (can be preprocessed like _NO3, _PO4, _Si)
        plotter: OceanMapPlotter instance for preprocessing
    """
    run_dir = Path(basedir) / run_name

    # Determine which file type to use based on variable
    # Diagnostic variables are in diad_T.nc
    # Tracer variables (nutrients, PFTs) are in ptrc_T.nc
    diagnostic_vars = ['Cflx', 'TChl', '_TChl', 'PPT', 'EXP', '_EXP', '_PPINT']

    if var_name in diagnostic_vars:
        file_type = 'diad_T'
    else:
        file_type = 'ptrc_T'

    # Try different filename patterns
    # Pattern 1: ORCA2_1m_YYYY0101_YYYY1231_<type>.nc
    nc_file = run_dir / f"ORCA2_1m_{year}0101_{year}1231_{file_type}.nc"

    # Pattern 2: run_name_YYYY0101_YYYY1231_<type>.nc (fallback)
    if not nc_file.exists():
        nc_file = run_dir / f"{run_name}_{year}0101_{year}1231_{file_type}.nc"

    if not nc_file.exists():
        print(f"Warning: File not found: {nc_file}")
        return None

    try:
        # Use plotter.load_data() to get preprocessed variables with unit conversions
        # Pass volume to create integrated variables (_PICINT, _BACINT, etc.)
        ds = plotter.load_data(str(nc_file), volume=plotter.volume)

        if var_name not in ds:
            print(f"Warning: Variable {var_name} not found after preprocessing")
            return None

        # Calculate annual mean
        data = ds[var_name].mean(dim='time_counter')

        # Take surface level if 3D
        if 'deptht' in data.dims or 'nav_lev' in data.dims:
            depth_dim = 'deptht' if 'deptht' in data.dims else 'nav_lev'
            data = data.isel({depth_dim: 0})

        # Apply land mask
        data = plotter.apply_mask(data)

        return data

    except Exception as e:
        print(f"Error loading {var_name} from {nc_file}: {e}")
        return None


def create_map_ax(fig, position, projection=ccrs.Robinson()):
    """Create a single map axis with cartopy features."""
    ax = fig.add_subplot(position, projection=projection)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.set_global()
    return ax


def plot_multimodel_maps(models, output_dir, config):
    """
    Create multi-panel spatial comparison maps.

    Args:
        models: List of dicts with 'name', 'desc', 'year', 'basedir'
        output_dir: Output directory
        config: Configuration dict
    """
    n_models = len(models)
    has_anomaly = (n_models == 2)
    n_cols = n_models + (1 if has_anomaly else 0)

    # Get DPI and format from config
    dpi = config.get("figure", {}).get("dpi", 300) if config else 300
    fmt = config.get("figure", {}).get("format", "png") if config else "png"

    # Create OceanMapPlotter for data preprocessing
    plotter = OceanMapPlotter()

    # Define variable groups - get metadata from map_utils for consistency
    # Use derived variables (with underscore) for integrated/processed values
    var_groups = {
        'ecosystem': ['_TChl', '_EXP', '_PPINT'],
        'nutrients': ['_PO4', '_NO3', '_Si', '_Fer'],
        'phytoplankton': ['_PICINT', '_FIXINT', '_COCINT', '_DIAINT', '_MIXINT', '_PHAINT'],
        'zooplankton': ['_BACINT', '_PROINT', '_MESINT', '_PTEINT', '_CRUINT', '_GELINT'],
    }

    projection = ccrs.PlateCarree()
    data_crs = ccrs.PlateCarree()

    # Load one file to get navigation
    nav_lon, nav_lat = None, None
    for model in models:
        run_dir = Path(model['basedir']) / model['name']
        ptrc_file = run_dir / f"ORCA2_1m_{model['year']}0101_{model['year']}1231_ptrc_T.nc"
        if not ptrc_file.exists():
            ptrc_file = run_dir / f"{model['name']}_{model['year']}0101_{model['year']}1231_ptrc_T.nc"
        if ptrc_file.exists():
            ds = xr.open_dataset(ptrc_file)
            nav_lon = ds.nav_lon if 'nav_lon' in ds else ds.lon
            nav_lat = ds.nav_lat if 'nav_lat' in ds else ds.lat
            ds.close()
            break

    if nav_lon is None:
        print("Error: Could not load navigation from any model files")
        return

    for group_name, var_names in var_groups.items():
        n_rows = len(var_names)

        # Create figure with subplots using constrained_layout
        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True)
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

        print(f"Generating {group_name} comparison map...")

        for row_idx, var_name in enumerate(var_names):
            # Get variable metadata from map_utils
            metadata = get_variable_metadata(var_name)
            var_label = metadata.get('long_name', var_name)
            var_unit = metadata.get('units', '')
            cmap = metadata.get('cmap', 'viridis')
            vmax = metadata.get('vmax', None)

            # Load data for all models
            model_data = []

            for model in models:
                data = load_model_data(
                    model['basedir'], model['name'], model['year'], var_name, plotter
                )
                model_data.append(data)

            # Skip if no data loaded
            if all(d is None for d in model_data):
                print(f"  Skipping {var_name}: no data available")
                continue

            # Determine color range
            if vmax is not None:
                if 'Cflx' in var_name or 'flux' in var_label.lower():
                    # Symmetric for flux variables
                    vrange = (-vmax, vmax)
                else:
                    vrange = (0, vmax)
            else:
                # Auto-determine from data
                valid_data = [d for d in model_data if d is not None]
                if valid_data:
                    all_vals = np.concatenate([d.values.flatten() for d in valid_data])
                    all_vals = all_vals[~np.isnan(all_vals)]
                    if 'Cflx' in var_name or 'flux' in var_label.lower():
                        # Symmetric for flux
                        vmax_calc = np.abs(np.percentile(all_vals, [5, 95])).max()
                        vrange = (-vmax_calc, vmax_calc)
                    else:
                        vrange = (np.percentile(all_vals, 5), np.percentile(all_vals, 95))

            # Plot each model
            for col_idx, (model, data) in enumerate(zip(models, model_data)):
                ax = create_map_ax(fig, gs[row_idx, col_idx], projection)

                if data is not None:
                    im = ax.pcolormesh(
                        nav_lon, nav_lat, data,
                        transform=data_crs,
                        cmap=cmap,
                        vmin=vrange[0], vmax=vrange[1],
                        shading='auto'
                    )

                    # Add colorbar below the map
                    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                                       pad=0.05, shrink=0.8)
                    cbar.set_label(var_unit, fontsize=12)
                    cbar.ax.tick_params(labelsize=10)

                # Add title (model name on first row, variable on first column)
                if row_idx == 0:
                    ax.set_title(model['name'], fontsize=14, fontweight='bold')
                if col_idx == 0:
                    ax.text(-0.1, 0.5, var_label, transform=ax.transAxes,
                           fontsize=14, fontweight='bold', rotation=90,
                           va='center', ha='right')

            # Add anomaly column if 2 models
            if has_anomaly and model_data[0] is not None and model_data[1] is not None:
                ax = create_map_ax(fig, gs[row_idx, n_cols-1], projection)

                diff = model_data[1] - model_data[0]  # B - A

                # Use percentiles to avoid outliers dominating the colorbar
                diff_vals = diff.values.flatten()
                diff_vals = diff_vals[~np.isnan(diff_vals)]
                if len(diff_vals) > 0:
                    # Use 95th percentile of absolute values for symmetric range
                    diff_max = np.percentile(np.abs(diff_vals), 95)
                else:
                    diff_max = 1.0  # Fallback

                im = ax.pcolormesh(
                    nav_lon, nav_lat, diff,
                    transform=data_crs,
                    cmap='RdBu_r',
                    vmin=-diff_max, vmax=diff_max,
                    shading='auto'
                )

                cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                                   pad=0.05, shrink=0.8)
                cbar.set_label(f'Δ {var_unit}', fontsize=12)
                cbar.ax.tick_params(labelsize=10)

                if row_idx == 0:
                    ax.set_title('Anomaly (B - A)', fontsize=14, fontweight='bold')

        # Save figure
        output_file = output_dir / f"multimodel_spatial_{group_name}.{fmt}"
        fig.savefig(output_file, dpi=dpi)
        print(f"Created {output_file}")
        plt.close(fig)


def main():
    if len(sys.argv) < 3:
        print("Usage: python multimodel_maps.py <models_csv> <output_dir>")
        return 1

    csv_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    # Load config
    config = load_config()

    # Read models from CSV (columns: model_id, description, start_year, to_year, location)
    models = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header row
        for row in reader:
            if len(row) >= 5:  # Ensure we have all columns
                models.append({
                    'name': row[0],      # model_id
                    'desc': row[1],      # description
                    'year': row[3],      # to_year
                    'basedir': row[4]    # location
                })

    print(f"Generating spatial comparison maps for {len(models)} models...")
    plot_multimodel_maps(models, output_dir, config)

    # Import transect functions
    try:
        from multimodel_transects import (
            plot_multimodel_nutrient_transects,
            plot_multimodel_pft_transects
        )

        print(f"\nGenerating nutrient transect comparisons...")
        plot_multimodel_nutrient_transects(models, output_dir, config)

        print(f"\nGenerating PFT transect comparisons...")
        plot_multimodel_pft_transects(models, output_dir, config)
    except ImportError as e:
        print(f"Warning: Could not import transect functions: {e}")
        print("Skipping transect generation. Run multimodel_transects.py separately if needed.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
