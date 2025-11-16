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


def load_model_data(basedir, run_name, year, var_name):
    """Load annual mean surface data for a variable."""
    ptrc_file = Path(basedir) / run_name / f"{run_name}_{year}0101_{year}1231_ptrc_T.nc"

    if not ptrc_file.exists():
        print(f"Warning: File not found: {ptrc_file}")
        return None, None, None

    try:
        ds = xr.open_dataset(ptrc_file)

        if var_name not in ds:
            print(f"Warning: Variable {var_name} not found")
            return None, None, None

        # Get navigation
        nav_lon = ds.nav_lon if 'nav_lon' in ds else ds.lon
        nav_lat = ds.nav_lat if 'nav_lat' in ds else ds.lat

        # Calculate annual mean
        data = ds[var_name].mean(dim='time_counter')

        # Take surface level if 3D
        if 'deptht' in data.dims or 'nav_lev' in data.dims:
            depth_dim = 'deptht' if 'deptht' in data.dims else 'nav_lev'
            data = data.isel({depth_dim: 0})

        ds.close()
        return data, nav_lon, nav_lat

    except Exception as e:
        print(f"Error loading {var_name} from {ptrc_file}: {e}")
        return None, None, None


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

    # Define variable groups
    var_groups = {
        'ecosystem': [
            ('Cflx', 'Air-Sea CO₂ Flux', 'mol C/m²/s', 'RdBu_r', None),
            ('TChl', 'Total Chlorophyll', 'mg Chl/m³', 'viridis', (0, 2)),
        ],
        'nutrients': [
            ('_PO4', 'Phosphate', 'mmol/m³', 'YlOrRd', (0, 3)),
            ('_NO3', 'Nitrate', 'mmol/m³', 'YlOrRd', (0, 40)),
            ('_Si', 'Silicate', 'mmol/m³', 'YlOrRd', (0, 100)),
        ],
    }

    projection = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    for group_name, variables in var_groups.items():
        n_rows = len(variables)

        # Create figure with subplots
        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.15)

        print(f"Generating {group_name} comparison map...")

        for row_idx, (var_name, var_label, var_unit, cmap, vrange) in enumerate(variables):
            # Load data for all models
            model_data = []
            nav_lon, nav_lat = None, None

            for model in models:
                data, lon, lat = load_model_data(
                    model['basedir'], model['name'], model['year'], var_name
                )
                model_data.append(data)
                if nav_lon is None and lon is not None:
                    nav_lon, nav_lat = lon, lat

            # Skip if no data loaded
            if all(d is None for d in model_data):
                print(f"  Skipping {var_name}: no data available")
                continue

            # Determine color range if not specified
            if vrange is None:
                valid_data = [d for d in model_data if d is not None]
                if valid_data:
                    all_vals = np.concatenate([d.values.flatten() for d in valid_data])
                    all_vals = all_vals[~np.isnan(all_vals)]
                    if 'Cflx' in var_name:
                        # Symmetric for flux
                        vmax = np.abs(np.percentile(all_vals, [5, 95])).max()
                        vrange = (-vmax, vmax)
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
                    cbar.set_label(var_unit, fontsize=8)

                # Add title (model name on first row, variable on first column)
                if row_idx == 0:
                    ax.set_title(model['desc'].replace('_', ' '), fontsize=10, fontweight='bold')
                if col_idx == 0:
                    ax.text(-0.1, 0.5, var_label, transform=ax.transAxes,
                           fontsize=10, fontweight='bold', rotation=90,
                           va='center', ha='right')

            # Add anomaly column if 2 models
            if has_anomaly and model_data[0] is not None and model_data[1] is not None:
                ax = create_map_ax(fig, gs[row_idx, n_cols-1], projection)

                diff = model_data[0] - model_data[1]
                diff_max = np.abs(diff).max().values

                im = ax.pcolormesh(
                    nav_lon, nav_lat, diff,
                    transform=data_crs,
                    cmap='RdBu_r',
                    vmin=-diff_max, vmax=diff_max,
                    shading='auto'
                )

                cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                                   pad=0.05, shrink=0.8)
                cbar.set_label(f'Δ {var_unit}', fontsize=8)

                if row_idx == 0:
                    ax.set_title('Anomaly (A - B)', fontsize=10, fontweight='bold')

        # Save figure
        output_file = output_dir / f"multimodel_spatial_{group_name}.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
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

    # Read models from CSV
    models = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append({
                'name': row['run'],
                'desc': row['description'],
                'year': row['to'],
                'basedir': row['location']
            })

    print(f"Generating spatial comparison maps for {len(models)} models...")
    plot_multimodel_maps(models, output_dir, config)

    return 0


if __name__ == '__main__':
    sys.exit(main())
