#!/usr/bin/env python3
"""
Generate difference/anomaly maps for 2-model comparisons.
Creates Model1 - Model2 difference maps for spatial diagnostics.

Usage:
    python make_difference_maps.py <model1_name> <model2_name> <year> <model1_model_dir> <model2_model_dir> <output_dir>
"""

import argparse
import sys
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Import map utilities from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from map_utils import OceanMapPlotter, get_variable_metadata


def load_annual_mean(model_dir, model_id, year, var_name):
    """Load annual mean for a specific variable."""
    ptrc_file = Path(model_dir) / model_id / f"{model_id}_{year}0101_{year}1231_ptrc_T.nc"

    if not ptrc_file.exists():
        print(f"Warning: File not found: {ptrc_file}")
        return None

    try:
        ds = xr.open_dataset(ptrc_file)
        if var_name not in ds:
            print(f"Warning: Variable {var_name} not found in {ptrc_file}")
            return None

        # Calculate annual mean
        data = ds[var_name].mean(dim='time_counter')
        return data
    except Exception as e:
        print(f"Error loading {var_name} from {ptrc_file}: {e}")
        return None


def calculate_surface_difference(data1, data2):
    """Calculate surface (top level) difference between two datasets."""
    if data1 is None or data2 is None:
        return None

    # Take surface level if 3D
    if 'deptht' in data1.dims or 'nav_lev' in data1.dims:
        depth_dim = 'deptht' if 'deptht' in data1.dims else 'nav_lev'
        surf1 = data1.isel({depth_dim: 0})
        surf2 = data2.isel({depth_dim: 0})
    else:
        surf1 = data1
        surf2 = data2

    # Calculate difference
    diff = surf1 - surf2
    return diff


def plot_difference_panel(plotter, differences, var_names, titles, output_path):
    """Plot a panel of difference maps."""
    n_vars = len(var_names)
    ncols = 2
    nrows = (n_vars + 1) // 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(12, 4 * nrows),
        subplot_kw={'projection': plotter.projection}
    )
    axes = axes.flatten() if n_vars > 1 else [axes]

    for idx, (var, title) in enumerate(zip(var_names, titles)):
        if var not in differences or differences[var] is None:
            axes[idx].text(0.5, 0.5, f'{var}\nData not available',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(title)
            continue

        diff_data = differences[var]

        # Plot difference with diverging colormap
        plotter.add_map_features(axes[idx])
        im = axes[idx].pcolormesh(
            plotter.nav_lon, plotter.nav_lat, diff_data,
            transform=plotter.data_crs,
            cmap='RdBu_r',  # Red for positive, Blue for negative
            vmin=-np.abs(diff_data).max(), vmax=np.abs(diff_data).max(),
            shading='auto'
        )
        plt.colorbar(im, ax=axes[idx], orientation='horizontal', pad=0.05, shrink=0.8)
        axes[idx].set_title(title)

    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Created difference map: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate model difference maps')
    parser.add_argument('model1', help='First model name')
    parser.add_argument('model2', help='Second model name')
    parser.add_argument('year', help='Year to compare')
    parser.add_argument('model_dir1', help='Base directory for model 1')
    parser.add_argument('model_dir2', help='Base directory for model 2')
    parser.add_argument('output_dir', help='Output directory')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Initialize plotter (need to load navigation from one of the models)
    ptrc_file = Path(args.model_dir1) / args.model1 / f"{args.model1}_{args.year}0101_{args.year}1231_ptrc_T.nc"
    if not ptrc_file.exists():
        print(f"Error: Cannot find model file: {ptrc_file}")
        return 1

    ds = xr.open_dataset(ptrc_file)
    plotter = OceanMapPlotter(ds.nav_lon, ds.nav_lat)

    # Generate diagnostics difference map (cflx, tchl, ppint, exp)
    print(f"Generating diagnostics difference map ({args.model1} - {args.model2})...")
    diag_vars = {
        'Cflx': ('Cflx', 'Air-Sea Carbon Flux Difference [mol C/m²/s]'),
        'TChl': ('TChl', 'Total Chlorophyll Difference [mg Chl/m³]'),
    }

    differences = {}
    for var, (var_name, title) in diag_vars.items():
        data1 = load_annual_mean(args.model_dir1, args.model1, args.year, var_name)
        data2 = load_annual_mean(args.model_dir2, args.model2, args.year, var_name)
        differences[var] = calculate_surface_difference(data1, data2)

    if any(diff is not None for diff in differences.values()):
        plot_difference_panel(
            plotter,
            differences,
            list(diag_vars.keys()),
            [title for _, title in diag_vars.values()],
            output_dir / f"difference_{args.year}_diagnostics.png"
        )

    ds.close()
    print(f"Difference maps complete for {args.model1} - {args.model2}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
