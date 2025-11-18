#!/usr/bin/env python3
"""
Generate multi-model transect comparison plots.
Creates vertical transect plots for nutrients and PFTs with anomaly plots.

For 2 models: columns = Model A, Model B, Anomaly (A-B)
For N models: columns = Model 1, Model 2, ..., Model N

Usage:
    python make_multimodel_transects.py <models_csv> <output_dir>
"""

import sys
import csv
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import map utilities from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from map_utils import (
    OceanMapPlotter,
    get_variable_metadata,
    PHYTOS, ZOOS,
    PHYTO_NAMES, ZOO_NAMES
)

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


def get_longitude_transect(data, nav_lon, target_lon, lat_values):
    """
    Extract data along a specific longitude transect.

    Args:
        data: xarray.DataArray to extract from
        nav_lon: 2D longitude array
        target_lon: Target longitude in degrees
        lat_values: Latitude values for the y dimension

    Returns:
        Data along the longitude transect
    """
    lon_diff = np.abs(nav_lon - target_lon)
    x_indices = lon_diff.argmin(dim='x')
    transect_data = data.isel(x=x_indices)
    transect_data['y'] = lat_values
    transect_data = transect_data.sortby('y')
    return transect_data


def get_central_latitude(nav_lat):
    """Extract latitude from the central meridian of a 2D nav_lat array"""
    mid_x = nav_lat.shape[1] // 2
    central_lat = nav_lat[:, mid_x]
    return central_lat


def load_transect_data(model_dir, model_id, year, variable, plotter):
    """
    Load 3D variable data for transect plotting.

    Args:
        model_dir: Base directory for model output
        model_id: Model run name
        year: Year to load
        variable: Variable name (e.g., 'PIC', 'BAC', '_NO3', '_PO4')
        plotter: OceanMapPlotter instance

    Returns:
        xarray.DataArray with depth-resolved data
    """
    run_dir = Path(model_dir) / model_id
    ptrc_file = run_dir / f"ORCA2_1m_{year}0101_{year}1231_ptrc_T.nc"

    if not ptrc_file.exists():
        print(f"Warning: File not found: {ptrc_file}")
        return None

    try:
        ds = xr.open_dataset(ptrc_file, decode_times=False)

        # Map derived variables (with underscore) to base variables and conversions
        variable_map = {
            '_NO3': ('NO3', 1e6),
            '_PO4': ('PO4', 1e6 / 122),
            '_Si': ('Si', 1e6),
            '_Fer': ('Fer', 1e9),
            '_O2': ('O2', 1e6),
        }

        # Determine the actual variable name and conversion factor
        if variable in variable_map:
            base_var, conversion = variable_map[variable]
        else:
            base_var = variable
            conversion = 1.0

        if base_var not in ds:
            print(f"Warning: Variable {base_var} not found")
            ds.close()
            return None

        # Time average
        data = ds[base_var].mean(dim='time_counter')

        # Apply unit conversion
        data = data * conversion

        # Remove bottom level if needed
        if 'deptht' in data.dims:
            data = data.isel(deptht=slice(None, -1))

        data = data.squeeze()
        ds.close()

        return data
    except Exception as e:
        print(f"Error loading {variable} from {ptrc_file}: {e}")
        return None


def plot_multimodel_nutrient_transects(models, output_dir, config, max_depth=None):
    """
    Create multi-model nutrient transect comparison plots.
    For 2 models: shows Model A, Model B, and Anomaly (B-A)
    For N models: shows all N models side by side

    Args:
        models: List of dicts with 'name', 'desc', 'year', 'model_dir'
        output_dir: Output directory
        config: Configuration dict
        max_depth: Maximum depth to plot in meters (default: None for full depth)
    """
    n_models = len(models)
    has_anomaly = (n_models == 2)
    n_cols = n_models + (1 if has_anomaly else 0)
    nutrients = ['_NO3', '_PO4', '_Si', '_Fer', '_O2']

    # Get DPI and format from config
    dpi = config.get("figure", {}).get("dpi", 300) if config else 300
    fmt = config.get("figure", {}).get("format", "png") if config else "png"

    # Create OceanMapPlotter
    plotter = OceanMapPlotter()

    # Load navigation from first available model
    nav_lon, nav_lat = None, None
    for model in models:
        run_dir = Path(model['model_dir']) / model['name']
        ptrc_file = run_dir / f"ORCA2_1m_{model['year']}0101_{model['year']}1231_ptrc_T.nc"
        if ptrc_file.exists():
            ds = xr.open_dataset(ptrc_file, decode_times=False)
            nav_lon = ds.nav_lon if 'nav_lon' in ds else ds.lon
            nav_lat = ds.nav_lat if 'nav_lat' in ds else ds.lat
            ds.close()
            break

    if nav_lon is None:
        print("Error: Could not load navigation from any model files")
        return

    lat_values = get_central_latitude(nav_lat.values)

    # Define transects
    transects = [
        ('Atlantic', -35.0, '35W'),
        ('Pacific', -170.0, '170W')
    ]

    for basin_name, target_lon, lon_label in transects:
        print(f"Generating {basin_name} nutrient transect comparison...")

        # Create figure: 5 nutrients (including O2) in rows, models+anomaly in columns
        n_nutrients = len(nutrients)

        # Use fig.subplots with sharex/sharey and constrained_layout
        fig, axs = plt.subplots(
            n_nutrients, n_cols,
            figsize=(4 * n_cols, 2.5 * n_nutrients),
            sharex='col',  # Share x-axis within each column
            sharey='row',  # Share y-axis within each row
            constrained_layout=True
        )

        for i, nut in enumerate(nutrients):
            # Load data for all models
            model_transects = []
            for model in models:
                data_3d = load_transect_data(
                    model['model_dir'], model['name'], model['year'], nut, plotter
                )

                if data_3d is not None:
                    transect = get_longitude_transect(data_3d, nav_lon, target_lon, lat_values)

                    # Limit depth to max_depth if specified
                    if max_depth is not None and 'deptht' in transect.coords:
                        depth_mask = transect.coords['deptht'] <= max_depth
                        transect = transect.where(depth_mask, drop=True)

                    model_transects.append(transect)
                else:
                    model_transects.append(None)

            # Use shared plotting function from difference_utils
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from difference_utils import plot_multimodel_transect_row

            model_labels = [m['name'] for m in models]

            plot_multimodel_transect_row(
                axs=axs[i, :],  # Pass the row of axes
                model_transects=model_transects,
                variable=nut,
                model_labels=model_labels,
                show_anomaly=has_anomaly,
                show_ylabel=(i == 0),  # Only first row gets full ylabel
                show_xlabel=(i == n_nutrients - 1),  # Only last row gets xlabel
                max_depth=max_depth
            )

        # Save
        output_file = output_dir / f"multimodel_transect_{basin_name.lower()}_nutrients.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Created {output_file}")
        plt.close(fig)


def plot_multimodel_pft_transects(models, output_dir, config, max_depth=500.0):
    """
    Create multi-model PFT transect comparison plots.
    For 2 models: shows Model A, Model B, and Anomaly (B-A)
    For N models: shows all N models side by side

    Args:
        models: List of dicts with 'name', 'desc', 'year', 'model_dir'
        output_dir: Output directory
        config: Configuration dict
        max_depth: Maximum depth to plot in meters (default: 500m)
    """
    n_models = len(models)
    has_anomaly = (n_models == 2)
    n_cols = n_models + (1 if has_anomaly else 0)
    pfts = PHYTOS + ZOOS

    # Get DPI and format from config
    dpi = config.get("figure", {}).get("dpi", 300) if config else 300
    fmt = config.get("figure", {}).get("format", "png") if config else "png"

    # Create OceanMapPlotter
    plotter = OceanMapPlotter()

    # Load navigation from first available model
    nav_lon, nav_lat = None, None
    for model in models:
        run_dir = Path(model['model_dir']) / model['name']
        ptrc_file = run_dir / f"ORCA2_1m_{model['year']}0101_{model['year']}1231_ptrc_T.nc"
        if ptrc_file.exists():
            ds = xr.open_dataset(ptrc_file, decode_times=False)
            nav_lon = ds.nav_lon if 'nav_lon' in ds else ds.lon
            nav_lat = ds.nav_lat if 'nav_lat' in ds else ds.lat
            ds.close()
            break

    if nav_lon is None:
        print("Error: Could not load navigation from any model files")
        return

    lat_values = get_central_latitude(nav_lat.values)

    # Define transects
    transects = [
        ('Atlantic', -35.0, '35W'),
        ('Pacific', -170.0, '170W')
    ]

    for basin_name, target_lon, lon_label in transects:
        print(f"Generating {basin_name} PFT transect comparison...")

        # Create figure: 12 PFTs in rows, models+anomaly in columns
        n_pfts = len(pfts)

        # Use fig.subplots with sharex/sharey and constrained_layout
        fig, axs = plt.subplots(
            n_pfts, n_cols,
            figsize=(4 * n_cols, 1.8 * n_pfts),
            sharex='col',  # Share x-axis within each column
            sharey='row',  # Share y-axis within each row
            constrained_layout=True
        )

        for i, pft in enumerate(pfts):
            # Load data for all models
            model_transects = []
            for model in models:
                data_3d = load_transect_data(
                    model['model_dir'], model['name'], model['year'], pft, plotter
                )

                if data_3d is not None:
                    transect = get_longitude_transect(data_3d, nav_lon, target_lon, lat_values)

                    # Convert from mmol C/m³ to µmol C/L (multiply by 1e6)
                    transect = transect * 1e6

                    # Limit depth to max_depth
                    if 'deptht' in transect.coords:
                        depth_mask = transect.coords['deptht'] <= max_depth
                        transect = transect.where(depth_mask, drop=True)

                    model_transects.append(transect)
                else:
                    model_transects.append(None)

            # Use shared plotting function from difference_utils
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from difference_utils import plot_multimodel_transect_row

            model_labels = [m['name'] for m in models]

            plot_multimodel_transect_row(
                axs=axs[i, :],  # Pass the row of axes
                model_transects=model_transects,
                variable=pft,
                model_labels=model_labels,
                show_anomaly=has_anomaly,
                show_ylabel=(i == 0),  # Only first row gets full ylabel
                show_xlabel=(i == n_pfts - 1),  # Only last row gets xlabel
                max_depth=max_depth
            )

        # Save
        output_file = output_dir / f"multimodel_transect_{basin_name.lower()}_pfts.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Created {output_file}")
        plt.close(fig)


def main():
    if len(sys.argv) < 3:
        print("Usage: python make_multimodel_transects.py <models_csv> <output_dir>")
        return 1

    csv_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    # Load config
    config = load_config()

    # Read models from CSV (columns: model_id, description, start_year, to_year, [location])
    # location column is optional - defaults to ~/scratch/ModelRuns if not provided
    import os
    default_model_dir = os.path.expanduser("~/scratch/ModelRuns")

    models = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Read header row
        has_location = len(header) >= 5 and header[4].strip().lower() == 'location'

        for row in reader:
            if len(row) >= 4:  # Need at least first 4 columns
                model_dir = row[4] if len(row) >= 5 and row[4].strip() else default_model_dir
                models.append({
                    'name': row[0],      # model_id
                    'desc': row[1],      # description
                    'year': row[3],      # to_year
                    'model_dir': model_dir   # location (or default)
                })

    print(f"Generating transect comparisons for {len(models)} models...")

    print(f"\nGenerating nutrient transect comparisons...")
    plot_multimodel_nutrient_transects(models, output_dir, config)

    print(f"\nGenerating PFT transect comparisons...")
    plot_multimodel_pft_transects(models, output_dir, config)

    return 0


if __name__ == '__main__':
    sys.exit(main())
